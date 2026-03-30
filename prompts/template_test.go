package prompts

import (
	"context"
	"testing"
)

func TestPromptTemplate(t *testing.T) {
	tmpl := NewPromptTemplate("Tell me a joke about {topic}")

	if len(tmpl.InputVariables) != 1 || tmpl.InputVariables[0] != "topic" {
		t.Errorf("expected InputVariables [topic], got %v", tmpl.InputVariables)
	}

	result, err := tmpl.Format(map[string]any{"topic": "golang"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "Tell me a joke about golang" {
		t.Errorf("expected 'Tell me a joke about golang', got %q", result)
	}
}

func TestPromptTemplateMultipleVars(t *testing.T) {
	tmpl := NewPromptTemplate("Hello {name}, you are {age} years old")

	if len(tmpl.InputVariables) != 2 {
		t.Errorf("expected 2 input variables, got %d", len(tmpl.InputVariables))
	}

	result, err := tmpl.Format(map[string]any{"name": "Alice", "age": 30})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "Hello Alice, you are 30 years old" {
		t.Errorf("unexpected result: %q", result)
	}
}

func TestPromptTemplateMissingVar(t *testing.T) {
	tmpl := NewPromptTemplate("Hello {name}")
	_, err := tmpl.Format(map[string]any{})
	if err == nil {
		t.Error("expected error for missing variable")
	}
}

func TestPromptTemplatePartialVars(t *testing.T) {
	tmpl := NewPromptTemplate("Hello {name}, {greeting}").
		WithPartialVariables(map[string]any{"greeting": "how are you?"})

	result, err := tmpl.Format(map[string]any{"name": "Alice"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "Hello Alice, how are you?" {
		t.Errorf("unexpected result: %q", result)
	}
}

func TestPromptTemplateInvoke(t *testing.T) {
	tmpl := NewPromptTemplate("Question: {question}")
	result, err := tmpl.Invoke(context.Background(), map[string]any{"question": "why?"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "Question: why?" {
		t.Errorf("unexpected result: %q", result)
	}
}

func TestPromptTemplateBatch(t *testing.T) {
	tmpl := NewPromptTemplate("Item: {item}")
	results, err := tmpl.Batch(context.Background(), []map[string]any{
		{"item": "apple"},
		{"item": "banana"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0] != "Item: apple" || results[1] != "Item: banana" {
		t.Errorf("unexpected results: %v", results)
	}
}

func TestPromptTemplateGetName(t *testing.T) {
	tmpl := NewPromptTemplate("test")
	if tmpl.GetName() != "PromptTemplate" {
		t.Errorf("expected 'PromptTemplate', got %q", tmpl.GetName())
	}
	tmpl.WithName("MyTemplate")
	if tmpl.GetName() != "MyTemplate" {
		t.Errorf("expected 'MyTemplate', got %q", tmpl.GetName())
	}
}

func TestPromptTemplateStream(t *testing.T) {
	tmpl := NewPromptTemplate("Hello {name}!")

	iter, err := tmpl.Stream(context.Background(), map[string]any{"name": "World"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	chunk, ok, err := iter.Next()
	if err != nil {
		t.Fatalf("unexpected error from Next: %v", err)
	}
	if !ok {
		t.Fatal("expected a chunk")
	}
	if chunk != "Hello World!" {
		t.Errorf("unexpected chunk: %q", chunk)
	}

	_, ok, err = iter.Next()
	if err != nil {
		t.Fatalf("unexpected error on second Next: %v", err)
	}
	if ok {
		t.Error("expected stream to be done")
	}
}

func TestPromptTemplateStreamError(t *testing.T) {
	tmpl := NewPromptTemplate("{missing}")
	_, err := tmpl.Stream(context.Background(), map[string]any{})
	if err == nil {
		t.Error("expected error for missing variable in stream")
	}
}

func TestPromptTemplateBatchError(t *testing.T) {
	tmpl := NewPromptTemplate("Hello {name}!")
	_, err := tmpl.Batch(context.Background(), []map[string]any{{}})
	if err == nil {
		t.Error("expected error for missing variable in batch")
	}
}
