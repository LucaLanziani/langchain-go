// Package llms provides the interfaces for language model integrations.
package llms

import (
	"context"

	"github.com/langchain-go/langchain-go/core"
)

// ChatModel is the interface that all chat model implementations must satisfy.
// It extends the Runnable interface with chat-specific methods.
type ChatModel interface {
	core.Runnable[[]core.Message, *core.AIMessage]

	// Generate performs a chat completion and returns detailed results
	// including token usage.
	Generate(ctx context.Context, messages []core.Message, opts ...core.Option) (*ChatResult, error)

	// BindTools returns a new ChatModel that will use the given tool definitions
	// when generating responses.
	BindTools(tools ...ToolDefinition) ChatModel

	// WithStructuredOutput configures the model to return structured output
	// matching the given JSON schema.
	WithStructuredOutput(schema map[string]any) ChatModel
}

// ToolDefinition describes a tool that can be bound to a chat model.
type ToolDefinition struct {
	// Name of the tool.
	Name string `json:"name"`

	// Description of what the tool does.
	Description string `json:"description"`

	// Parameters is a JSON Schema describing the tool's parameters.
	Parameters map[string]any `json:"parameters"`
}
