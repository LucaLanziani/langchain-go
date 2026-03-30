package llms

import (
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
)

func TestLLMOptions(t *testing.T) {
	cfg := core.ApplyOptions(
		WithTemperature(0.7),
		WithMaxTokens(100),
		WithTopP(0.9),
		WithModel("gpt-4"),
	)

	if temp, ok := cfg.Configurable[ConfigKeyTemperature].(float64); !ok || temp != 0.7 {
		t.Errorf("expected temperature=0.7, got %v", cfg.Configurable[ConfigKeyTemperature])
	}
	if max, ok := cfg.Configurable[ConfigKeyMaxTokens].(int); !ok || max != 100 {
		t.Errorf("expected max_tokens=100, got %v", cfg.Configurable[ConfigKeyMaxTokens])
	}
	if top, ok := cfg.Configurable[ConfigKeyTopP].(float64); !ok || top != 0.9 {
		t.Errorf("expected top_p=0.9, got %v", cfg.Configurable[ConfigKeyTopP])
	}
	if model, ok := cfg.Configurable[ConfigKeyModel].(string); !ok || model != "gpt-4" {
		t.Errorf("expected model='gpt-4', got %v", cfg.Configurable[ConfigKeyModel])
	}
}

func TestChatResultStructure(t *testing.T) {
	result := &ChatResult{
		Generations: []*ChatGeneration{
			{GenerationInfo: map[string]any{"tokens": 10}},
		},
		LLMOutput: map[string]any{"model": "gpt-4"},
	}
	if len(result.Generations) != 1 {
		t.Errorf("expected 1 generation, got %d", len(result.Generations))
	}
}

func TestTokenUsage(t *testing.T) {
	usage := TokenUsage{
		PromptTokens:     10,
		CompletionTokens: 20,
		TotalTokens:      30,
	}
	if usage.TotalTokens != 30 {
		t.Errorf("expected 30 total tokens, got %d", usage.TotalTokens)
	}
}

func TestToolDefinition(t *testing.T) {
	def := ToolDefinition{
		Name:        "test",
		Description: "A test tool",
		Parameters:  map[string]any{"type": "object"},
	}
	if def.Name != "test" {
		t.Errorf("expected name 'test', got %q", def.Name)
	}
}
