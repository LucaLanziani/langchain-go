package outputparsers

import (
	"context"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
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
	fresult, ferr := parser.ParseMessage(msg)
	if ferr != nil {
		t.Fatalf("unexpected error: %v", ferr)
	}
	if fresult.Name != "Dave" {
		t.Errorf("expected name 'Dave', got %q", fresult.Name)
	}
}

func TestJSONOutputParserWithName(t *testing.T) {
	parser := NewJSONOutputParser[testStruct]()
	if parser.GetName() != "JSONOutputParser" {
		t.Errorf("expected 'JSONOutputParser', got %q", parser.GetName())
	}
	parser.WithName("Custom")
	if parser.GetName() != "Custom" {
		t.Errorf("expected 'Custom', got %q", parser.GetName())
	}
}

func TestJSONOutputParserGetFormatInstructions(t *testing.T) {
	parser := NewJSONOutputParser[testStruct]()
	instructions := parser.GetFormatInstructions()
	if instructions == "" {
		t.Error("expected non-empty format instructions")
	}
}

func TestJSONOutputParserStream(t *testing.T) {
	parser := NewJSONOutputParser[testStruct]()
	msg := core.NewAIMessage(`{"name": "Eve", "age": 28}`)

	iter, err := parser.Stream(context.Background(), msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	chunk, ok, err := iter.Next()
	if err != nil {
		t.Fatalf("unexpected iter error: %v", err)
	}
	if !ok {
		t.Fatal("expected a chunk")
	}
	if chunk.Name != "Eve" {
		t.Errorf("unexpected name: %q", chunk.Name)
	}

	_, ok, err = iter.Next()
	if err != nil {
		t.Fatalf("unexpected error on second Next: %v", err)
	}
	if ok {
		t.Error("expected stream to be done")
	}
}

func TestJSONOutputParserStreamError(t *testing.T) {
	parser := NewJSONOutputParser[testStruct]()
	msg := core.NewAIMessage("invalid json")
	_, err := parser.Stream(context.Background(), msg)
	if err == nil {
		t.Error("expected error for invalid JSON in stream")
	}
}

func TestJSONOutputParserBatch(t *testing.T) {
	parser := NewJSONOutputParser[testStruct]()
	results, err := parser.Batch(context.Background(), []*core.AIMessage{
		core.NewAIMessage(`{"name": "Alice", "age": 30}`),
		core.NewAIMessage(`{"name": "Bob", "age": 25}`),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Name != "Alice" || results[1].Name != "Bob" {
		t.Errorf("unexpected results: %v", results)
	}
}

func TestJSONOutputParserBatchError(t *testing.T) {
	parser := NewJSONOutputParser[testStruct]()
	_, err := parser.Batch(context.Background(), []*core.AIMessage{
		core.NewAIMessage(`{"name": "Alice", "age": 30}`),
		core.NewAIMessage("bad json"),
	})
	if err == nil {
		t.Error("expected error for invalid JSON in batch")
	}
}
