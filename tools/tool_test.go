package tools

import (
	"context"
	"testing"

	"github.com/langchain-go/langchain-go/core"
)

func TestNewTool(t *testing.T) {
	tool := NewTool("greet", "Greets a person", func(_ context.Context, input string) (string, error) {
		return "Hello, " + input, nil
	})

	if tool.Name() != "greet" {
		t.Errorf("expected name 'greet', got %q", tool.Name())
	}
	if tool.Description() != "Greets a person" {
		t.Errorf("expected description 'Greets a person', got %q", tool.Description())
	}

	result, err := tool.Run(context.Background(), "Alice")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "Hello, Alice" {
		t.Errorf("expected 'Hello, Alice', got %q", result)
	}
}

type searchArgs struct {
	Query string `json:"query" description:"The search query"`
	Limit int    `json:"limit,omitempty" description:"Max results"`
}

func TestNewTypedTool(t *testing.T) {
	tool := NewTypedTool("search", "Search the web", searchArgs{},
		func(_ context.Context, args searchArgs) (string, error) {
			return "results for: " + args.Query, nil
		},
	)

	if tool.Name() != "search" {
		t.Errorf("expected name 'search', got %q", tool.Name())
	}

	// Test schema generation.
	schema := tool.ArgsSchema()
	if schema["type"] != "object" {
		t.Errorf("expected schema type 'object', got %v", schema["type"])
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties map")
	}
	queryProp, ok := props["query"].(map[string]any)
	if !ok {
		t.Fatal("expected query property")
	}
	if queryProp["type"] != "string" {
		t.Errorf("expected query type 'string', got %v", queryProp["type"])
	}
	if queryProp["description"] != "The search query" {
		t.Errorf("expected description, got %v", queryProp["description"])
	}

	// Test execution with JSON input.
	result, err := tool.Run(context.Background(), `{"query": "golang", "limit": 10}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "results for: golang" {
		t.Errorf("expected 'results for: golang', got %q", result)
	}
}

func TestToDefinitions(t *testing.T) {
	tool := NewTool("test", "A test tool", func(_ context.Context, input string) (string, error) {
		return input, nil
	})
	defs := ToDefinitions(tool)
	if len(defs) != 1 {
		t.Fatalf("expected 1 definition, got %d", len(defs))
	}
	if defs[0].Name != "test" {
		t.Errorf("expected name 'test', got %q", defs[0].Name)
	}
}

func TestExecuteToolCall(t *testing.T) {
	tool := NewTool("calc", "Calculator", func(_ context.Context, input string) (string, error) {
		return "42", nil
	})

	tc := core.ToolCall{ID: "call_1", Name: "calc", Args: []byte(`{"input": "2+2"}`)}
	result, err := ExecuteToolCall(context.Background(), tc, []Tool{tool})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "42" {
		t.Errorf("expected '42', got %q", result)
	}
}

func TestExecuteToolCallNotFound(t *testing.T) {
	tc := core.ToolCall{ID: "call_1", Name: "missing"}
	_, err := ExecuteToolCall(context.Background(), tc, nil)
	if err == nil {
		t.Error("expected error for missing tool")
	}
}

func TestRunnableTool(t *testing.T) {
	tool := NewTool("echo", "Echoes input", func(_ context.Context, input string) (string, error) {
		return input, nil
	})
	rt := NewRunnableTool(tool)
	result, err := rt.Invoke(context.Background(), "test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "test" {
		t.Errorf("expected 'test', got %q", result)
	}
}
