package core

import (
	"context"
)

// CallbackHandler is the interface for receiving events during LangChain execution.
// Implementations can override any subset of methods. The default no-op
// implementations are provided by BaseCallbackHandler.
type CallbackHandler interface {
	// LLM callbacks
	OnLLMStart(ctx context.Context, prompts []string, runID string, parentRunID string, extras map[string]any)
	OnChatModelStart(ctx context.Context, messages []Message, runID string, parentRunID string, extras map[string]any)
	OnLLMNewToken(ctx context.Context, token string, runID string)
	OnLLMEnd(ctx context.Context, output *LLMResult, runID string)
	OnLLMError(ctx context.Context, err error, runID string)

	// Chain callbacks
	OnChainStart(ctx context.Context, inputs map[string]any, runID string, parentRunID string, extras map[string]any)
	OnChainEnd(ctx context.Context, outputs map[string]any, runID string)
	OnChainError(ctx context.Context, err error, runID string)

	// Tool callbacks
	OnToolStart(ctx context.Context, toolName string, input string, runID string, parentRunID string)
	OnToolEnd(ctx context.Context, output string, runID string)
	OnToolError(ctx context.Context, err error, runID string)

	// Agent callbacks
	OnAgentAction(ctx context.Context, action AgentActionData, runID string)
	OnAgentFinish(ctx context.Context, finish AgentFinishData, runID string)

	// Retriever callbacks
	OnRetrieverStart(ctx context.Context, query string, runID string, parentRunID string)
	OnRetrieverEnd(ctx context.Context, documents []*Document, runID string)
	OnRetrieverError(ctx context.Context, err error, runID string)

	// Text callbacks
	OnText(ctx context.Context, text string, runID string)
}

// AgentActionData holds data for agent action callbacks.
type AgentActionData struct {
	Tool      string `json:"tool"`
	ToolInput string `json:"tool_input"`
	Log       string `json:"log"`
}

// AgentFinishData holds data for agent finish callbacks.
type AgentFinishData struct {
	Output map[string]any `json:"output"`
	Log    string         `json:"log"`
}

// LLMResult holds the result of an LLM call for callbacks.
type LLMResult struct {
	Generations []string       `json:"generations"`
	LLMOutput   map[string]any `json:"llm_output,omitempty"`
}

// BaseCallbackHandler provides no-op implementations of all CallbackHandler methods.
// Embed this in your handler to only override the methods you care about.
type BaseCallbackHandler struct{}

func (BaseCallbackHandler) OnLLMStart(_ context.Context, _ []string, _ string, _ string, _ map[string]any) {
}
func (BaseCallbackHandler) OnChatModelStart(_ context.Context, _ []Message, _ string, _ string, _ map[string]any) {
}
func (BaseCallbackHandler) OnLLMNewToken(_ context.Context, _ string, _ string)               {}
func (BaseCallbackHandler) OnLLMEnd(_ context.Context, _ *LLMResult, _ string)                {}
func (BaseCallbackHandler) OnLLMError(_ context.Context, _ error, _ string)                   {}
func (BaseCallbackHandler) OnChainStart(_ context.Context, _ map[string]any, _ string, _ string, _ map[string]any) {
}
func (BaseCallbackHandler) OnChainEnd(_ context.Context, _ map[string]any, _ string)          {}
func (BaseCallbackHandler) OnChainError(_ context.Context, _ error, _ string)                 {}
func (BaseCallbackHandler) OnToolStart(_ context.Context, _ string, _ string, _ string, _ string) {
}
func (BaseCallbackHandler) OnToolEnd(_ context.Context, _ string, _ string)                   {}
func (BaseCallbackHandler) OnToolError(_ context.Context, _ error, _ string)                  {}
func (BaseCallbackHandler) OnAgentAction(_ context.Context, _ AgentActionData, _ string)      {}
func (BaseCallbackHandler) OnAgentFinish(_ context.Context, _ AgentFinishData, _ string)      {}
func (BaseCallbackHandler) OnRetrieverStart(_ context.Context, _ string, _ string, _ string)  {}
func (BaseCallbackHandler) OnRetrieverEnd(_ context.Context, _ []*Document, _ string)         {}
func (BaseCallbackHandler) OnRetrieverError(_ context.Context, _ error, _ string)             {}
func (BaseCallbackHandler) OnText(_ context.Context, _ string, _ string)                      {}

// Ensure BaseCallbackHandler implements CallbackHandler.
var _ CallbackHandler = (*BaseCallbackHandler)(nil)
