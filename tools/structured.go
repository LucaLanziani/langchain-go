package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

// StructuredTool is a tool created from a Go function with typed arguments.
type StructuredTool struct {
	name        string
	description string
	argsSchema  map[string]any
	fn          func(ctx context.Context, input string) (string, error)
}

// Name returns the tool name.
func (t *StructuredTool) Name() string { return t.name }

// Description returns the tool description.
func (t *StructuredTool) Description() string { return t.description }

// ArgsSchema returns the JSON Schema for the tool's parameters.
func (t *StructuredTool) ArgsSchema() map[string]any { return t.argsSchema }

// Run executes the tool.
func (t *StructuredTool) Run(ctx context.Context, input string) (string, error) {
	return t.fn(ctx, input)
}

// NewTool creates a StructuredTool from a name, description, and function.
// The function receives the raw JSON string input and returns a string result.
func NewTool(name, description string, fn func(ctx context.Context, input string) (string, error)) *StructuredTool {
	return &StructuredTool{
		name:        name,
		description: description,
		argsSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"input": map[string]any{
					"type":        "string",
					"description": "The input to the tool",
				},
			},
			"required": []string{"input"},
		},
		fn: fn,
	}
}

// NewTypedTool creates a StructuredTool with typed input.
// The argsType should be a struct with json tags that defines the input schema.
// The function receives the parsed struct as JSON and returns a string result.
//
// Example:
//
//	type SearchArgs struct {
//	    Query string `json:"query" description:"The search query"`
//	}
//	tool := NewTypedTool("search", "Search the web", SearchArgs{},
//	    func(ctx context.Context, args SearchArgs) (string, error) {
//	        return "results for: " + args.Query, nil
//	    },
//	)
func NewTypedTool[T any](name, description string, argsExample T, fn func(ctx context.Context, args T) (string, error)) *StructuredTool {
	schema := generateJSONSchema(argsExample)
	return &StructuredTool{
		name:        name,
		description: description,
		argsSchema:  schema,
		fn: func(ctx context.Context, input string) (string, error) {
			var args T
			if err := json.Unmarshal([]byte(input), &args); err != nil {
				return "", fmt.Errorf("failed to parse tool input: %w", err)
			}
			return fn(ctx, args)
		},
	}
}

// generateJSONSchema generates a JSON Schema from a Go struct.
func generateJSONSchema(v any) map[string]any {
	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	if t.Kind() != reflect.Struct {
		return map[string]any{
			"type": "object",
			"properties": map[string]any{
				"input": map[string]any{"type": "string"},
			},
		}
	}

	properties := make(map[string]any)
	var required []string

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}

		jsonTag := field.Tag.Get("json")
		if jsonTag == "-" {
			continue
		}

		name := field.Name
		omitempty := false
		if jsonTag != "" {
			parts := strings.Split(jsonTag, ",")
			name = parts[0]
			for _, p := range parts[1:] {
				if p == "omitempty" {
					omitempty = true
				}
			}
		}

		prop := map[string]any{}

		// Determine JSON Schema type from Go type.
		switch field.Type.Kind() {
		case reflect.String:
			prop["type"] = "string"
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			prop["type"] = "integer"
		case reflect.Float32, reflect.Float64:
			prop["type"] = "number"
		case reflect.Bool:
			prop["type"] = "boolean"
		case reflect.Slice:
			prop["type"] = "array"
			elemType := goTypeToJSONType(field.Type.Elem().Kind())
			prop["items"] = map[string]any{"type": elemType}
		default:
			prop["type"] = "object"
		}

		// Use description tag if available.
		if desc := field.Tag.Get("description"); desc != "" {
			prop["description"] = desc
		}

		properties[name] = prop

		if !omitempty {
			required = append(required, name)
		}
	}

	schema := map[string]any{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

// goTypeToJSONType maps Go reflect.Kind to JSON Schema type string.
func goTypeToJSONType(k reflect.Kind) string {
	switch k {
	case reflect.String:
		return "string"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return "integer"
	case reflect.Float32, reflect.Float64:
		return "number"
	case reflect.Bool:
		return "boolean"
	default:
		return "object"
	}
}

// Ensure StructuredTool implements Tool.
var _ Tool = (*StructuredTool)(nil)
