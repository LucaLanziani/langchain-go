package outputparsers

import (
	"context"
	"testing"

	"github.com/langchain-go/langchain-go/core"
)

func TestStringOutputParser(t *testing.T) {
	parser := NewStringOutputParser()

	msg := core.NewAIMessage("Hello, world!")
	result, err := parser.Parse(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "Hello, world!" {
		t.Errorf("expected 'Hello, world!', got %q", result)
	}
}

func TestStringOutputParserInvoke(t *testing.T) {
	parser := NewStringOutputParser()
	msg := core.NewAIMessage("test output")
	result, err := parser.Invoke(context.Background(), msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "test output" {
		t.Errorf("expected 'test output', got %q", result)
	}
}

func TestStringOutputParserBatch(t *testing.T) {
	parser := NewStringOutputParser()
	results, err := parser.Batch(context.Background(), []*core.AIMessage{
		core.NewAIMessage("a"),
		core.NewAIMessage("b"),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 || results[0] != "a" || results[1] != "b" {
		t.Errorf("unexpected results: %v", results)
	}
}

func TestStringOutputParserParseMessage(t *testing.T) {
	parser := NewStringOutputParser()
	var msg core.Message = core.NewAIMessage("via interface")
	result, err := parser.ParseMessage(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "via interface" {
		t.Errorf("expected 'via interface', got %q", result)
	}
}

func TestStringOutputParserGetName(t *testing.T) {
	parser := NewStringOutputParser()
	if parser.GetName() != "StringOutputParser" {
		t.Errorf("expected 'StringOutputParser', got %q", parser.GetName())
	}
	parser.WithName("Custom")
	if parser.GetName() != "Custom" {
		t.Errorf("expected 'Custom', got %q", parser.GetName())
	}
}
