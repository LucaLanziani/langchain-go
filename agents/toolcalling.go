package agents

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/langchain-go/langchain-go/core"
	"github.com/langchain-go/langchain-go/llms"
	"github.com/langchain-go/langchain-go/prompts"
	"github.com/langchain-go/langchain-go/tools"
)

// ToolCallingAgent uses a chat model's native tool calling capability.
// This is the modern, recommended agent type.
type ToolCallingAgent struct {
	llm    llms.ChatModel
	prompt *prompts.ChatPromptTemplate
	tools  []tools.Tool
}

// NewToolCallingAgent creates a new ToolCallingAgent.
// The prompt must include a MessagesPlaceholder("agent_scratchpad") for intermediate steps.
func NewToolCallingAgent(llm llms.ChatModel, agentTools []tools.Tool, prompt *prompts.ChatPromptTemplate) *ToolCallingAgent {
	// Bind tools to the model.
	toolDefs := tools.ToDefinitions(agentTools...)
	boundLLM := llm.BindTools(toolDefs...)

	return &ToolCallingAgent{
		llm:    boundLLM,
		prompt: prompt,
		tools:  agentTools,
	}
}

// Plan decides the next action(s) based on intermediate steps and inputs.
func (a *ToolCallingAgent) Plan(ctx context.Context, intermediateSteps []AgentStep, inputs map[string]any) (*AgentOutput, error) {
	// Build the agent scratchpad from intermediate steps.
	scratchpad := formatToolCallingSteps(intermediateSteps)

	// Merge inputs with scratchpad.
	mergedInputs := make(map[string]any)
	for k, v := range inputs {
		mergedInputs[k] = v
	}
	mergedInputs["agent_scratchpad"] = scratchpad

	// Format prompt.
	messages, err := a.prompt.FormatMessages(mergedInputs)
	if err != nil {
		return nil, fmt.Errorf("failed to format prompt: %w", err)
	}

	// Call the model.
	response, err := a.llm.Invoke(ctx, messages)
	if err != nil {
		return nil, fmt.Errorf("LLM call failed: %w", err)
	}

	// If the model returned tool calls, create actions.
	if len(response.ToolCalls) > 0 {
		actions := make([]AgentAction, len(response.ToolCalls))
		for i, tc := range response.ToolCalls {
			actions[i] = AgentAction{
				Tool:      tc.Name,
				ToolInput: string(tc.Args),
				Log:       fmt.Sprintf("Calling tool: %s", tc.Name),
				MessageLog: []core.Message{response},
			}
		}
		return &AgentOutput{Actions: actions}, nil
	}

	// No tool calls = agent is done.
	return &AgentOutput{
		Finish: &AgentFinish{
			ReturnValues: map[string]any{
				"output": response.Content,
			},
			Log:        response.Content,
			MessageLog: []core.Message{response},
		},
	}, nil
}

// InputKeys returns the expected input keys.
func (a *ToolCallingAgent) InputKeys() []string {
	// Filter out agent_scratchpad from prompt variables.
	var keys []string
	for _, v := range a.prompt.InputVariables {
		if v != "agent_scratchpad" {
			keys = append(keys, v)
		}
	}
	return keys
}

// OutputKeys returns the output keys.
func (a *ToolCallingAgent) OutputKeys() []string {
	return []string{"output"}
}

// formatToolCallingSteps converts intermediate steps to messages for the scratchpad.
func formatToolCallingSteps(steps []AgentStep) []core.Message {
	var messages []core.Message
	for _, step := range steps {
		// Add the AI message that triggered this tool call.
		argsJSON := step.Action.ToolInput
		// Ensure valid JSON for args.
		if !json.Valid([]byte(argsJSON)) {
			argsJSON = fmt.Sprintf(`{"input": %q}`, argsJSON)
		}

		// Create an AI message with the tool call.
		aiMsg := core.NewAIMessageWithToolCalls("", []core.ToolCall{
			{
				ID:   fmt.Sprintf("call_%s", step.Action.Tool),
				Name: step.Action.Tool,
				Args: json.RawMessage(argsJSON),
				Type: "function",
			},
		})
		messages = append(messages, aiMsg)

		// Add the tool result message.
		messages = append(messages, core.NewToolMessage(
			step.Observation,
			fmt.Sprintf("call_%s", step.Action.Tool),
		))
	}
	return messages
}

// Ensure ToolCallingAgent implements Agent.
var _ Agent = (*ToolCallingAgent)(nil)
