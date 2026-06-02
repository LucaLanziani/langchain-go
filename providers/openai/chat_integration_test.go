//go:build integration

package openai

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	integrationtest "github.com/LucaLanziani/langchain-go/internal/integrationtest"
)

func TestLMStudioOpenAIInvoke(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	response, err := newLMStudioOpenAIModel().Invoke(ctx, lmStudioTestMessages())
	if err != nil {
		t.Fatalf("openai-compatible LM Studio invoke failed: %v", err)
	}
	if !integrationtest.HasOutput(response) {
		t.Fatalf("expected content or tool call response, got content=%q tool_calls=%d", response.GetContent(), len(response.ToolCalls))
	}

	t.Logf("openai-compatible LM Studio invoke response: content=%q tool_calls=%d", response.GetContent(), len(response.ToolCalls))
}

func TestLMStudioOpenAIStream(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	stream, err := newLMStudioOpenAIModel().Stream(ctx, lmStudioTestMessages())
	if err != nil {
		t.Fatalf("openai-compatible LM Studio stream failed: %v", err)
	}
	defer stream.Close()

	chunks, err := stream.Collect()
	if err != nil {
		t.Fatalf("openai-compatible LM Studio stream collection failed: %v", err)
	}

	content, toolCalls := integrationtest.StreamSummary(chunks)
	if strings.TrimSpace(content) == "" && toolCalls == 0 {
		t.Fatalf("expected streamed content or tool calls, got chunks=%d", len(chunks))
	}

	t.Logf("openai-compatible LM Studio stream response: content=%q tool_calls=%d", content, toolCalls)
}

func newLMStudioOpenAIModel() *ChatModel {
	return New(
		WithBaseURL(integrationtest.BaseURL("LMSTUDIO_OPENAI_BASE_URL", "/v1")),
		WithAPIKey(integrationtest.AuthToken("LMSTUDIO_OPENAI_AUTH_TOKEN")),
		WithModelName(integrationtest.Model("LMSTUDIO_OPENAI_MODEL")),
	)
}

func lmStudioTestMessages() []core.Message {
	return []core.Message{
		core.NewSystemMessage("You are a concise assistant."),
		core.NewHumanMessage("Reply with a short greeting."),
	}
}
