package outputparsers

import (
	"context"
	"testing"

	"github.com/langchain-go/langchain-go/core"
)

type testStruct struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

func TestJSONOutputParser(t *testing.T) {
	parser := NewJSONOutputParser[testStruct]()
	msg := core.NewAIMessage(`{"name": "Alice", "age": 30}`)

	result, err := parser.Parse(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Name != "Alice" || result.Age != 30 {
		t.Errorf("unexpected result: %+v", result)
	}
}

func TestJSONOutputParserCodeBlock(t *testing.T) {
	parser := NewJSONOutputParser[testStruct]()
	msg := core.NewAIMessage("Here is the result:\n```json\n{\"name\": \"Bob\", \"age\": 25}\n```")

	result, err := parser.Parse(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Name != "Bob" || result.Age != 25 {
		t.Errorf("unexpected result: %+v", result)
	}
}

func TestJSONOutputParserMap(t *testing.T) {
	parser := NewJSONOutputParser[map[string]any]()
	msg := core.NewAIMessage(`{"key": "value", "num": 42}`)

	result, err := parser.Parse(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["key"] != "value" {
		t.Errorf("expected key=value, got %v", result["key"])
	}
}

func TestJSONOutputParserInvoke(t *testing.T) {
	parser := NewJSONOutputParser[testStruct]()
	msg := core.NewAIMessage(`{"name": "Charlie", "age": 35}`)

	result, err := parser.Invoke(context.Background(), msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Name != "Charlie" {
		t.Errorf("expected name 'Charlie', got %q", result.Name)
	}
}

func TestJSONOutputParserInvalidJSON(t *testing.T) {
	parser := NewJSONOutputParser[testStruct]()
	msg := core.NewAIMessage("not json at all")

	_, err := parser.Parse(msg)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestJSONOutputParserParseMessage(t *testing.T) {
	parser := NewJSONOutputParser[testStruct]()
	var msg core.Message = core.NewAIMessage(`{"name": "Dave", "age": 40}`)
	result, err := parser.ParseMessage(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Name != "Dave" {
		t.Errorf("expected name 'Dave', got %q", result.Name)
	}
}
