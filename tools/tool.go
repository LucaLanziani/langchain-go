// Package tools provides the tool interface and utilities for creating tools
// that can be used by LangChain agents.
package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
)

// Tool is the interface that all tools must implement.
// Tools are functions that agents can call to interact with the world.
type Tool interface {
	// Name returns the unique name of the tool.
	Name() string

	// Description returns a description of what the tool does.
	// This is used by the model to decide when to use the tool.
	Description() string

	// ArgsSchema returns a JSON Schema describing the tool's parameters.
	ArgsSchema() map[string]any

	// Run executes the tool with the given input string or JSON.
	Run(ctx context.Context, input string) (string, error)
}

// ToDefinition converts a Tool to an llms.ToolDefinition for model binding.
func ToDefinition(t Tool) llms.ToolDefinition {
	return llms.ToolDefinition{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters:  t.ArgsSchema(),
	}
}

// ToDefinitions converts multiple tools to ToolDefinitions.
func ToDefinitions(tools ...Tool) []llms.ToolDefinition {
	defs := make([]llms.ToolDefinition, len(tools))
	for i, t := range tools {
		defs[i] = ToDefinition(t)
	}
	return defs
}

// RunnableTool wraps a Tool as a core.Runnable[string, string].
type RunnableTool struct {
	tool Tool
}

// NewRunnableTool wraps a Tool as a Runnable.
func NewRunnableTool(t Tool) *RunnableTool {
	return &RunnableTool{tool: t}
}

// GetName returns the tool name.
func (r *RunnableTool) GetName() string { return r.tool.Name() }

// Invoke runs the tool.
func (r *RunnableTool) Invoke(ctx context.Context, input string, opts ...core.Option) (string, error) {
	return r.tool.Run(ctx, input)
}

// Stream returns a single-chunk stream.
func (r *RunnableTool) Stream(ctx context.Context, input string, opts ...core.Option) (*core.StreamIterator[string], error) {
	result, err := r.tool.Run(ctx, input)
	if err != nil {
		return nil, err
	}
	ch := make(chan core.StreamChunk[string], 1)
	ch <- core.StreamChunk[string]{Value: result}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

// Batch runs the tool for multiple inputs.
func (r *RunnableTool) Batch(ctx context.Context, inputs []string, opts ...core.Option) ([]string, error) {
	results := make([]string, len(inputs))
	for i, input := range inputs {
		result, err := r.tool.Run(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// ExecuteToolCall executes a tool call from an AI message, looking up the tool by name.
func ExecuteToolCall(ctx context.Context, toolCall core.ToolCall, availableTools []Tool) (string, error) {
	for _, t := range availableTools {
		if t.Name() == toolCall.Name {
			return t.Run(ctx, string(toolCall.Args))
		}
	}
	return "", fmt.Errorf("tool %q not found", toolCall.Name)
}

// ExecuteToolCalls executes all tool calls from an AI message.
func ExecuteToolCalls(ctx context.Context, toolCalls []core.ToolCall, availableTools []Tool) ([]core.Message, error) {
	var results []core.Message
	for _, tc := range toolCalls {
		output, err := ExecuteToolCall(ctx, tc, availableTools)
		if err != nil {
			// Return error as a tool message so the agent can see it.
			output = fmt.Sprintf("Error: %v", err)
		}
		results = append(results, core.NewToolMessage(output, tc.ID))
	}
	return results, nil
}

// ParseToolCallArgs parses the JSON args of a tool call into the given struct.
func ParseToolCallArgs(tc core.ToolCall, v any) error {
	if err := json.Unmarshal(tc.Args, v); err != nil {
		return fmt.Errorf("failed to parse tool call args for %s: %w", tc.Name, err)
	}
	return nil
}
