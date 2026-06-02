// Package llms provides the interfaces for language model integrations.
package llms

import (
	"context"

	"github.com/LucaLanziani/langchain-go/core"
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

	// BindSkills returns a new ChatModel that will use the given skill
	// definitions when generating responses. Binding a skill does not guarantee
	// provider-side emission in the first iteration; providers without native
	// skill handling ignore bound skills silently.
	BindSkills(skills ...SkillDefinition) ChatModel

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

// SkillDefinition describes a reusable provider-native skill that can be bound
// to a chat model. Binding a skill does not guarantee provider-side emission in
// the first iteration; providers without native skill support ignore bound
// skills silently.
type SkillDefinition struct {
	// Name of the skill.
	Name string `json:"name"`

	// Description of what the skill does.
	Description string `json:"description"`

	// Instructions contains provider-neutral guidance associated with the skill.
	Instructions string `json:"instructions"`

	// Parameters is an optional JSON Schema describing the skill's expected
	// inputs for providers that support structured skill configuration.
	Parameters map[string]any `json:"parameters"`
}
