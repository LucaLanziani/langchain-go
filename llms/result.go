package llms

import (
	"github.com/langchain-go/langchain-go/core"
)

// ChatResult holds the complete result of a chat model invocation.
type ChatResult struct {
	// Generations contains the generated messages.
	Generations []*ChatGeneration `json:"generations"`

	// LLMOutput contains provider-specific output data.
	LLMOutput map[string]any `json:"llm_output,omitempty"`
}

// ChatGeneration represents a single generated message.
type ChatGeneration struct {
	// Message is the generated AI message.
	Message *core.AIMessage `json:"message"`

	// GenerationInfo contains generation-specific metadata.
	GenerationInfo map[string]any `json:"generation_info,omitempty"`
}

// TokenUsage tracks token consumption.
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
