package agents

import (
	"context"
	"fmt"
	"strings"

	"github.com/langchain-go/langchain-go/core"
	"github.com/langchain-go/langchain-go/tools"
)

// Agent is the interface for the planning component that decides what to do next.
type Agent interface {
	// Plan takes the intermediate steps so far and returns the next action(s) or finish.
	Plan(ctx context.Context, intermediateSteps []AgentStep, inputs map[string]any) (*AgentOutput, error)

	// InputKeys returns the expected input keys.
	InputKeys() []string

	// OutputKeys returns the output keys.
	OutputKeys() []string
}

// AgentExecutor runs an agent loop: plan -> execute tool -> plan -> ... -> finish.
// It implements Runnable[map[string]any, map[string]any].
type AgentExecutor struct {
	agent         Agent
	tools         []tools.Tool
	toolMap       map[string]tools.Tool
	maxIterations int
	returnIntermediateSteps bool
	handleParsingErrors     bool
	name                    string
	callbacks               []core.CallbackHandler
}

// NewAgentExecutor creates a new AgentExecutor.
func NewAgentExecutor(agent Agent, agentTools []tools.Tool, options ...ExecutorOption) *AgentExecutor {
	toolMap := make(map[string]tools.Tool)
	for _, t := range agentTools {
		toolMap[t.Name()] = t
	}

	exec := &AgentExecutor{
		agent:         agent,
		tools:         agentTools,
		toolMap:       toolMap,
		maxIterations: 15,
	}

	for _, opt := range options {
		opt(exec)
	}

	return exec
}

// ExecutorOption configures the AgentExecutor.
type ExecutorOption func(*AgentExecutor)

// WithMaxIterations sets the maximum number of agent loop iterations.
func WithMaxIterations(n int) ExecutorOption {
	return func(e *AgentExecutor) { e.maxIterations = n }
}

// WithReturnIntermediateSteps includes intermediate steps in the output.
func WithReturnIntermediateSteps(v bool) ExecutorOption {
	return func(e *AgentExecutor) { e.returnIntermediateSteps = v }
}

// WithHandleParsingErrors enables handling of parsing errors.
func WithHandleParsingErrors(v bool) ExecutorOption {
	return func(e *AgentExecutor) { e.handleParsingErrors = v }
}

// GetName returns the executor name.
func (e *AgentExecutor) GetName() string {
	if e.name != "" {
		return e.name
	}
	return "AgentExecutor"
}

// Invoke runs the agent loop to completion.
func (e *AgentExecutor) Invoke(ctx context.Context, input map[string]any, opts ...core.Option) (map[string]any, error) {
	cfg := core.ApplyOptions(opts...)

	// Notify callbacks.
	for _, cb := range cfg.Callbacks {
		cb.OnChainStart(ctx, input, cfg.RunID, "", map[string]any{"name": e.GetName()})
	}

	var intermediateSteps []AgentStep
	iterations := 0

	for iterations < e.maxIterations {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		output, err := e.agent.Plan(ctx, intermediateSteps, input)
		if err != nil {
			if e.handleParsingErrors {
				intermediateSteps = append(intermediateSteps, AgentStep{
					Action:      AgentAction{Tool: "_error", ToolInput: "", Log: err.Error()},
					Observation: fmt.Sprintf("Error: %v. Please try again with valid output.", err),
				})
				iterations++
				continue
			}
			for _, cb := range cfg.Callbacks {
				cb.OnChainError(ctx, err, cfg.RunID)
			}
			return nil, fmt.Errorf("agent planning error: %w", err)
		}

		// Agent decided to finish.
		if output.Finish != nil {
			result := output.Finish.ReturnValues
			if e.returnIntermediateSteps {
				result["intermediate_steps"] = intermediateSteps
			}
			for _, cb := range cfg.Callbacks {
				cb.OnAgentFinish(ctx, core.AgentFinishData{
					Output: result,
					Log:    output.Finish.Log,
				}, cfg.RunID)
				cb.OnChainEnd(ctx, result, cfg.RunID)
			}
			return result, nil
		}

		// Execute the tool calls.
		for _, action := range output.Actions {
			for _, cb := range cfg.Callbacks {
				cb.OnAgentAction(ctx, core.AgentActionData{
					Tool:      action.Tool,
					ToolInput: action.ToolInput,
					Log:       action.Log,
				}, cfg.RunID)
			}

			tool, ok := e.toolMap[action.Tool]
			if !ok {
				observation := fmt.Sprintf("Tool %q not found. Available tools: %s",
					action.Tool, e.availableToolNames())
				intermediateSteps = append(intermediateSteps, AgentStep{
					Action:      action,
					Observation: observation,
				})
				continue
			}

			for _, cb := range cfg.Callbacks {
				cb.OnToolStart(ctx, action.Tool, action.ToolInput, cfg.RunID, "")
			}

			observation, err := tool.Run(ctx, action.ToolInput)
			if err != nil {
				observation = fmt.Sprintf("Error executing tool %s: %v", action.Tool, err)
				for _, cb := range cfg.Callbacks {
					cb.OnToolError(ctx, err, cfg.RunID)
				}
			} else {
				for _, cb := range cfg.Callbacks {
					cb.OnToolEnd(ctx, observation, cfg.RunID)
				}
			}

			intermediateSteps = append(intermediateSteps, AgentStep{
				Action:      action,
				Observation: observation,
			})
		}

		iterations++
	}

	err := fmt.Errorf("agent exceeded maximum iterations (%d)", e.maxIterations)
	for _, cb := range cfg.Callbacks {
		cb.OnChainError(ctx, err, cfg.RunID)
	}
	return nil, err
}

// Stream runs the agent and returns a single-chunk stream with the final output.
func (e *AgentExecutor) Stream(ctx context.Context, input map[string]any, opts ...core.Option) (*core.StreamIterator[map[string]any], error) {
	result, err := e.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	ch := make(chan core.StreamChunk[map[string]any], 1)
	ch <- core.StreamChunk[map[string]any]{Value: result}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

// Batch runs the agent for multiple inputs.
func (e *AgentExecutor) Batch(ctx context.Context, inputs []map[string]any, opts ...core.Option) ([]map[string]any, error) {
	results := make([]map[string]any, len(inputs))
	for i, input := range inputs {
		result, err := e.Invoke(ctx, input, opts...)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

func (e *AgentExecutor) availableToolNames() string {
	names := make([]string, len(e.tools))
	for i, t := range e.tools {
		names[i] = t.Name()
	}
	return strings.Join(names, ", ")
}
