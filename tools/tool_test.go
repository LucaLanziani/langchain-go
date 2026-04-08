package tools

import (
	"context"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
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

func TestRunnableToolGetName(t *testing.T) {
	tool := NewTool("mytool", "desc", func(_ context.Context, input string) (string, error) {
		return input, nil
	})
	rt := NewRunnableTool(tool)
	if rt.GetName() != "mytool" {
		t.Errorf("expected 'mytool', got %q", rt.GetName())
	}
}

func TestRunnableToolStream(t *testing.T) {
	tool := NewTool("stream_tool", "desc", func(_ context.Context, input string) (string, error) {
		return "streamed:" + input, nil
	})
	rt := NewRunnableTool(tool)
	iter, err := rt.Stream(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	result, ok, err := iter.Next()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Fatal("expected a chunk")
	}
	if result != "streamed:hello" {
		t.Errorf("expected 'streamed:hello', got %q", result)
	}
}

func TestRunnableToolBatch(t *testing.T) {
	tool := NewTool("batch_tool", "desc", func(_ context.Context, input string) (string, error) {
		return "out:" + input, nil
	})
	rt := NewRunnableTool(tool)
	results, err := rt.Batch(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if results[0] != "out:a" || results[1] != "out:b" {
		t.Errorf("unexpected results: %v", results)
	}
}

func TestExecuteToolCalls(t *testing.T) {
	tool := NewTool("adder", "Adds", func(_ context.Context, input string) (string, error) {
		return "result:" + input, nil
	})
	toolCalls := []core.ToolCall{
		{ID: "c1", Name: "adder", Args: []byte(`{}`)},
		{ID: "c2", Name: "nonexistent", Args: []byte(`{}`)},
	}
	messages, err := ExecuteToolCalls(context.Background(), toolCalls, []Tool{tool})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}
}

func TestParseToolCallArgs(t *testing.T) {
	type myArgs struct {
		Query string `json:"query"`
	}
	tc := core.ToolCall{ID: "c1", Name: "search", Args: []byte(`{"query":"golang"}`)}
	var args myArgs
	if err := ParseToolCallArgs(tc, &args); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if args.Query != "golang" {
		t.Errorf("expected 'golang', got %q", args.Query)
	}
}

func TestParseToolCallArgsInvalid(t *testing.T) {
	tc := core.ToolCall{ID: "c1", Name: "tool", Args: []byte(`not valid json`)}
	var args map[string]any
	err := ParseToolCallArgs(tc, &args)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

type typedSchemaTest struct {
	Name    string   `json:"name" description:"A name"`
	Count   int      `json:"count"`
	Score   float64  `json:"score"`
	Active  bool     `json:"active,omitempty"`
	Tags    []string `json:"tags"`
	Private string   `json:"-"`
}

func TestGenerateJSONSchemaAllTypes(t *testing.T) {
	tool := NewTypedTool("typed", "A typed tool", typedSchemaTest{},
		func(_ context.Context, args typedSchemaTest) (string, error) {
			return args.Name, nil
		},
	)
	schema := tool.ArgsSchema()
	props := schema["properties"].(map[string]any)

	// Check all type mappings.
	for field, expectedType := range map[string]string{
		"name": "string", "count": "integer", "score": "number", "tags": "array",
	} {
		p, ok := props[field].(map[string]any)
		if !ok {
			t.Errorf("field %q not in schema", field)
			continue
		}
		if p["type"] != expectedType {
			t.Errorf("field %q: expected type %q, got %v", field, expectedType, p["type"])
		}
	}
	// active has omitempty so it should not be in required.
	required, _ := schema["required"].([]string)
	for _, r := range required {
		if r == "active" {
			t.Error("active should not be required (omitempty)")
		}
	}
	// private should be excluded.
	if _, ok := props["Private"]; ok {
		t.Error("private field should not appear in schema")
	}
}

func TestGenerateJSONSchemaNonStruct(t *testing.T) {
	schema := generateJSONSchema("not a struct")
	if schema["type"] != "object" {
		t.Errorf("expected type 'object', got %v", schema["type"])
	}
}

func TestGenerateJSONSchemaBoolType(t *testing.T) {
	type boolArgs struct {
		Flag bool `json:"flag"`
	}
	tool := NewTypedTool("bt", "bool test", boolArgs{},
		func(_ context.Context, args boolArgs) (string, error) { return "", nil },
	)
	schema := tool.ArgsSchema()
	props := schema["properties"].(map[string]any)
	p := props["flag"].(map[string]any)
	if p["type"] != "boolean" {
		t.Errorf("expected 'boolean', got %v", p["type"])
	}
}

type nestedContact struct {
	Email string `json:"email" format:"email"`
}

type nestedSchemaArgs struct {
	User struct {
		Name    string        `json:"name"`
		Status  string        `json:"status" enum:"active,inactive"`
		Contact nestedContact `json:"contact"`
	} `json:"user"`
	Aliases []nestedContact `json:"aliases,omitempty"`
}

func TestGenerateJSONSchemaNestedStructs(t *testing.T) {
	tool := NewTypedTool("nested", "nested schema", nestedSchemaArgs{},
		func(_ context.Context, args nestedSchemaArgs) (string, error) { return args.User.Name, nil },
	)
	schema := tool.ArgsSchema()
	user := schema["properties"].(map[string]any)["user"].(map[string]any)
	userProps := user["properties"].(map[string]any)
	status := userProps["status"].(map[string]any)
	contact := userProps["contact"].(map[string]any)
	aliases := schema["properties"].(map[string]any)["aliases"].(map[string]any)

	if user["type"] != "object" {
		t.Fatalf("expected nested user schema to be object, got %v", user["type"])
	}
	if status["enum"].([]any)[0] != "active" {
		t.Fatalf("expected enum values to be preserved, got %v", status["enum"])
	}
	if contact["properties"].(map[string]any)["email"].(map[string]any)["format"] != "email" {
		t.Fatalf("expected nested format tag to be preserved")
	}
	if aliases["items"].(map[string]any)["properties"].(map[string]any)["email"].(map[string]any)["format"] != "email" {
		t.Fatalf("expected slice items to recurse into nested structs")
	}
}
