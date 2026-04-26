package openai

import (
	"errors"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
)

type failingStreamReader struct {
	payload []byte
	sent    bool
}

func (r *failingStreamReader) Read(p []byte) (int, error) {
	if !r.sent {
		r.sent = true
		return copy(p, r.payload), nil
	}
	return 0, errors.New("stream broke")
}

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

func TestBindSkillsDoesNotAliasDerivedModels(t *testing.T) {
	base := &ChatModel{opts: DefaultOptions(), boundSkills: make([]llms.SkillDefinition, 1, 4)}
	base.boundSkills[0] = llms.SkillDefinition{Name: "base"}

	left := base.BindSkills(llms.SkillDefinition{Name: "left"}).(*ChatModel)
	right := base.BindSkills(llms.SkillDefinition{Name: "right"}).(*ChatModel)

	if left.boundSkills[1].Name != "left" {
		t.Fatalf("expected left skill to stay isolated, got %q", left.boundSkills[1].Name)
	}
	if right.boundSkills[1].Name != "right" {
		t.Fatalf("expected right skill to stay isolated, got %q", right.boundSkills[1].Name)
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

func TestStreamResponsePropagatesScannerErrors(t *testing.T) {
	reader := &failingStreamReader{
		payload: []byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n"),
	}
	ch := make(chan core.StreamChunk[*core.AIMessage], 2)

	model := &ChatModel{}
	model.streamResponse(reader, ch)

	first := <-ch
	if first.Err != nil {
		t.Fatalf("unexpected first chunk error: %v", first.Err)
	}
	if first.Value == nil || first.Value.Content != "hi" {
		t.Fatalf("expected first chunk content 'hi', got %#v", first.Value)
	}

	second := <-ch
	if second.Err == nil {
		t.Fatal("expected scanner error chunk")
	}
}
