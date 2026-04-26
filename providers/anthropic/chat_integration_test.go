//go:build integration

package anthropic

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	integrationtest "github.com/LucaLanziani/langchain-go/internal/integrationtest"
)

func TestLMStudioAnthropicInvoke(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	response, err := newLMStudioAnthropicModel().Invoke(ctx, lmStudioTestMessages())
	if err != nil {
		t.Fatalf("anthropic-compatible LM Studio invoke failed: %v", err)
	}
	if !integrationtest.HasOutput(response) {
		t.Fatalf("expected content or tool call response, got content=%q tool_calls=%d", response.GetContent(), len(response.ToolCalls))
	}

	t.Logf("anthropic-compatible LM Studio invoke response: content=%q tool_calls=%d", response.GetContent(), len(response.ToolCalls))
}

func TestLMStudioAnthropicStream(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	stream, err := newLMStudioAnthropicModel().Stream(ctx, lmStudioTestMessages())
	if err != nil {
		t.Fatalf("anthropic-compatible LM Studio stream failed: %v", err)
	}
	defer stream.Close()

	chunks, err := stream.Collect()
	if err != nil {
		t.Fatalf("anthropic-compatible LM Studio stream collection failed: %v", err)
	}

	content, toolCalls := integrationtest.StreamSummary(chunks)
	if strings.TrimSpace(content) == "" && toolCalls == 0 {
		t.Fatalf("expected streamed content or tool calls, got chunks=%d", len(chunks))
	}

	t.Logf("anthropic-compatible LM Studio stream response: content=%q tool_calls=%d", content, toolCalls)
}

func newLMStudioAnthropicModel() *ChatModel {
	// LM Studio exposes the Anthropic-compatible API under /v1/messages.
	return New(
		WithBaseURL(integrationtest.BaseURL("LMSTUDIO_ANTHROPIC_BASE_URL", "/v1")),
		WithAPIKey(integrationtest.AuthToken("LMSTUDIO_ANTHROPIC_AUTH_TOKEN")),
		WithModelName(integrationtest.Model("LMSTUDIO_ANTHROPIC_MODEL")),
		WithMaxTokens(128),
	)
}

func lmStudioTestMessages() []core.Message {
	return []core.Message{
		core.NewSystemMessage("You are a concise assistant."),
		core.NewHumanMessage("Reply with a short greeting."),
	}
}
