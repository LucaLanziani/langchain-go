package anthropic

import (
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
)

func TestBindToolsDoesNotAliasDerivedModels(t *testing.T) {
	base := &ChatModel{opts: DefaultOptions(), boundTools: make([]llms.ToolDefinition, 1, 4)}
	base.boundTools[0] = llms.ToolDefinition{Name: "base"}

	left := base.BindTools(llms.ToolDefinition{Name: "left"}).(*ChatModel)
	right := base.BindTools(llms.ToolDefinition{Name: "right"}).(*ChatModel)

	if left.boundTools[1].Name != "left" {
		t.Fatalf("expected left tool to stay isolated, got %q", left.boundTools[1].Name)
	}
	if right.boundTools[1].Name != "right" {
		t.Fatalf("expected right tool to stay isolated, got %q", right.boundTools[1].Name)
	}
}

func TestWithStructuredOutputClonesSchema(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
	}

	model := (&ChatModel{opts: DefaultOptions()}).WithStructuredOutput(schema).(*ChatModel)
	schema["properties"].(map[string]any)["name"].(map[string]any)["type"] = "integer"

	got := model.structuredSchema["properties"].(map[string]any)["name"].(map[string]any)["type"]
	if got != "string" {
		t.Fatalf("expected cloned schema to remain unchanged, got %v", got)
	}
}

func TestBuildRequestRejectsIncompatibleThinkingTemperature(t *testing.T) {
	temp := 0.2
	model := &ChatModel{opts: &Options{Model: "claude", MaxTokens: 256, ThinkingBudget: 64, Temperature: &temp}}

	_, err := model.buildRequest([]core.Message{core.NewHumanMessage("hi")}, core.DefaultConfig(), false)
	if err == nil {
		t.Fatal("expected temperature validation error")
	}
}

func TestBuildRequestRejectsIncompatibleThinkingTopP(t *testing.T) {
	topP := 0.8
	model := &ChatModel{opts: &Options{Model: "claude", MaxTokens: 256, ThinkingBudget: 64, TopP: &topP}}

	_, err := model.buildRequest([]core.Message{core.NewHumanMessage("hi")}, core.DefaultConfig(), false)
	if err == nil {
		t.Fatal("expected top_p validation error")
	}
}

func TestResponseToMessagePreservesThinkingBlocks(t *testing.T) {
	model := &ChatModel{}
	msg := model.responseToMessage(&anthropicResponse{
		Content: []anthropicContent{
			{Type: "thinking", Text: "intermediate reasoning"},
			{Type: "text", Text: "final answer"},
		},
	})

	if msg.Content != "final answer" {
		t.Fatalf("expected final answer content, got %q", msg.Content)
	}
	if got := msg.AdditionalKwargs["thinking"]; got != "intermediate reasoning" {
		t.Fatalf("expected thinking block to be preserved, got %v", got)
	}
}
