package core

import (
	"context"
	"testing"
)

// embeddedHandler embeds BaseCallbackHandler but overrides nothing.
type embeddedHandler struct {
	BaseCallbackHandler
}

func TestBaseCallbackHandlerNoOps(t *testing.T) {
	h := &embeddedHandler{}
	ctx := context.Background()

	// Call each no-op method to ensure they don't panic.
	h.OnLLMStart(ctx, []string{"prompt"}, "run1", "", nil)
	h.OnChatModelStart(ctx, []Message{NewHumanMessage("hi")}, "run1", "", nil)
	h.OnLLMNewToken(ctx, "token", "run1")
	h.OnLLMEnd(ctx, &LLMResult{Generations: []string{"output"}}, "run1")
	h.OnLLMError(ctx, ErrTest, "run1")
	h.OnChainStart(ctx, map[string]any{"input": "x"}, "run1", "", nil)
	h.OnChainEnd(ctx, map[string]any{"output": "y"}, "run1")
	h.OnChainError(ctx, ErrTest, "run1")
	h.OnToolStart(ctx, "mytool", "input", "run1", "")
	h.OnToolEnd(ctx, "result", "run1")
	h.OnToolError(ctx, ErrTest, "run1")
	h.OnAgentAction(ctx, AgentActionData{Tool: "t", ToolInput: "i"}, "run1")
	h.OnAgentFinish(ctx, AgentFinishData{Output: map[string]any{"k": "v"}}, "run1")
	h.OnRetrieverStart(ctx, "query", "run1", "")
	h.OnRetrieverEnd(ctx, []*Document{{PageContent: "doc"}}, "run1")
	h.OnRetrieverError(ctx, ErrTest, "run1")
	h.OnText(ctx, "some text", "run1")
}
