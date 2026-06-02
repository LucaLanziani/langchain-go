package copilot

import (
	"context"
	"encoding/json"
	"testing"

	copilot "github.com/github/copilot-sdk/go"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
	"github.com/LucaLanziani/langchain-go/tools"
)

// mockTool is a simple test tool.
type mockTool struct {
	name        string
	description string
	schema      map[string]any
	runFunc     func(ctx context.Context, input string) (string, error)
}

func (m *mockTool) Name() string               { return m.name }
func (m *mockTool) Description() string        { return m.description }
func (m *mockTool) ArgsSchema() map[string]any { return m.schema }
func (m *mockTool) Run(ctx context.Context, input string) (string, error) {
	return m.runFunc(ctx, input)
}

func TestNew(t *testing.T) {
	// This test requires a real Copilot CLI setup, so we skip it in CI.
	// It's here as a template for manual testing.
	t.Skip("Requires GitHub Copilot CLI setup")

	ctx := context.Background()
	model, err := New(ctx, WithModelName("gpt-5-mini"))
	if err != nil {
		t.Fatalf("failed to create model: %v", err)
	}
	defer model.Close()

	if model.GetName() != "ChatGitHubCopilot" {
		t.Errorf("expected name 'ChatGitHubCopilot', got %q", model.GetName())
	}
}

func TestGetName(t *testing.T) {
	model := &ChatModel{}
	if model.GetName() != "ChatGitHubCopilot" {
		t.Errorf("expected default name 'ChatGitHubCopilot', got %q", model.GetName())
	}

	model.name = "CustomName"
	if model.GetName() != "CustomName" {
		t.Errorf("expected custom name 'CustomName', got %q", model.GetName())
	}
}

func TestBuildClientOptions(t *testing.T) {
	opts := DefaultOptions()
	opts.GithubToken = "test-token"
	opts.CLIPath = "/usr/local/bin/copilot"
	opts.LogLevel = "debug"

	clientOpts := buildClientOptions(opts)

	if clientOpts.LogLevel != "debug" {
		t.Fatalf("expected log level 'debug', got %q", clientOpts.LogLevel)
	}
	if clientOpts.GitHubToken != "test-token" {
		t.Fatalf("expected GitHub token to be forwarded, got %q", clientOpts.GitHubToken)
	}
	if clientOpts.CLIPath != "/usr/local/bin/copilot" {
		t.Fatalf("expected CLI path to be forwarded, got %q", clientOpts.CLIPath)
	}
	if len(clientOpts.CLIArgs) != 1 || clientOpts.CLIArgs[0] != "--disable-builtin-mcps" {
		t.Fatalf("expected built-in MCP restriction CLI args, got %v", clientOpts.CLIArgs)
	}
}

func TestBindTools(t *testing.T) {
	model := &ChatModel{opts: DefaultOptions()}

	tool1 := llms.ToolDefinition{Name: "tool1", Description: "First tool"}
	tool2 := llms.ToolDefinition{Name: "tool2", Description: "Second tool"}

	bound := model.BindTools(tool1, tool2)

	// Original model should be unchanged.
	if len(model.boundTools) != 0 {
		t.Errorf("original model should have 0 bound tools, got %d", len(model.boundTools))
	}

	// Bound model should have the tools.
	boundModel := bound.(*ChatModel)
	if len(boundModel.boundTools) != 2 {
		t.Errorf("expected 2 bound tools, got %d", len(boundModel.boundTools))
	}
	if boundModel.boundTools[0].Name != "tool1" {
		t.Errorf("expected first tool name 'tool1', got %q", boundModel.boundTools[0].Name)
	}
}

func TestBindToolsDoesNotAliasDerivedModels(t *testing.T) {
	model := &ChatModel{opts: DefaultOptions(), boundTools: make([]llms.ToolDefinition, 1, 4)}
	model.boundTools[0] = llms.ToolDefinition{Name: "base"}

	left := model.BindTools(llms.ToolDefinition{Name: "left"}).(*ChatModel)
	right := model.BindTools(llms.ToolDefinition{Name: "right"}).(*ChatModel)

	if left.boundTools[1].Name != "left" {
		t.Fatalf("expected isolated left tool, got %q", left.boundTools[1].Name)
	}
	if right.boundTools[1].Name != "right" {
		t.Fatalf("expected isolated right tool, got %q", right.boundTools[1].Name)
	}
}

func TestBindSkillsDoesNotAliasDerivedModels(t *testing.T) {
	model := &ChatModel{opts: DefaultOptions(), boundSkills: make([]llms.SkillDefinition, 1, 4)}
	model.boundSkills[0] = llms.SkillDefinition{Name: "base"}

	left := model.BindSkills(llms.SkillDefinition{Name: "left"}).(*ChatModel)
	right := model.BindSkills(llms.SkillDefinition{Name: "right"}).(*ChatModel)

	if left.boundSkills[1].Name != "left" {
		t.Fatalf("expected isolated left skill, got %q", left.boundSkills[1].Name)
	}
	if right.boundSkills[1].Name != "right" {
		t.Fatalf("expected isolated right skill, got %q", right.boundSkills[1].Name)
	}
}

func TestWithStructuredOutput(t *testing.T) {
	model := &ChatModel{opts: DefaultOptions()}

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
	}

	structured := model.WithStructuredOutput(schema)

	// Original model should be unchanged.
	if model.structuredSchema != nil {
		t.Error("original model should not have structured schema")
	}

	// Structured model should have the schema.
	structuredModel := structured.(*ChatModel)
	if structuredModel.structuredSchema == nil {
		t.Fatal("structured model should have schema")
	}
	if structuredModel.structuredSchema["type"] != "object" {
		t.Errorf("expected schema type 'object', got %v", structuredModel.structuredSchema["type"])
	}
}

func TestWithStructuredOutputClonesSchema(t *testing.T) {
	model := &ChatModel{opts: DefaultOptions()}
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
	}

	structuredModel := model.WithStructuredOutput(schema).(*ChatModel)
	schema["properties"].(map[string]any)["name"].(map[string]any)["type"] = "integer"

	got := structuredModel.structuredSchema["properties"].(map[string]any)["name"].(map[string]any)["type"]
	if got != "string" {
		t.Fatalf("expected cloned schema to remain unchanged, got %v", got)
	}
}

func TestExtractSystemMessage(t *testing.T) {
	tests := []struct {
		name     string
		messages []core.Message
		want     string
	}{
		{
			name:     "no messages",
			messages: []core.Message{},
			want:     "",
		},
		{
			name: "no system message",
			messages: []core.Message{
				core.NewHumanMessage("hello"),
				core.NewAIMessage("hi"),
			},
			want: "",
		},
		{
			name: "system message first",
			messages: []core.Message{
				core.NewSystemMessage("You are helpful"),
				core.NewHumanMessage("hello"),
			},
			want: "You are helpful",
		},
		{
			name: "system message in middle",
			messages: []core.Message{
				core.NewHumanMessage("hello"),
				core.NewSystemMessage("Be concise"),
				core.NewAIMessage("hi"),
			},
			want: "Be concise",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractSystemMessage(tt.messages)
			if got != tt.want {
				t.Errorf("extractSystemMessage() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestMessagesToPrompt(t *testing.T) {
	tests := []struct {
		name     string
		messages []core.Message
		want     string
	}{
		{
			name:     "empty messages",
			messages: []core.Message{},
			want:     "",
		},
		{
			name: "single human message",
			messages: []core.Message{
				core.NewHumanMessage("hello"),
			},
			want: "hello",
		},
		{
			name: "system message skipped",
			messages: []core.Message{
				core.NewSystemMessage("You are helpful"),
				core.NewHumanMessage("hello"),
			},
			want: "hello",
		},
		{
			name: "conversation",
			messages: []core.Message{
				core.NewHumanMessage("hello"),
				core.NewAIMessage("hi there"),
				core.NewHumanMessage("how are you?"),
			},
			want: "hello\nAssistant: hi there\nhow are you?",
		},
		{
			name: "with tool calls",
			messages: []core.Message{
				core.NewHumanMessage("what's the weather?"),
				core.NewAIMessageWithToolCalls("", []core.ToolCall{
					{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{"city":"SF"}`)},
				}),
				core.NewToolMessage("sunny", "call_1"),
			},
			want: "what's the weather?\nAssistant: \n[Tool Call: get_weather({\"city\":\"SF\"})]\n[Tool Result (call_1): sunny]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := messagesToPrompt(tt.messages)
			if got != tt.want {
				t.Errorf("messagesToPrompt() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestBuildSessionConfig(t *testing.T) {
	ctx := context.Background()

	t.Run("basic config", func(t *testing.T) {
		model := &ChatModel{
			opts: &Options{
				Model:           "gpt-5-mini",
				ReasoningEffort: "medium",
			},
		}

		messages := []core.Message{core.NewHumanMessage("hello")}
		cfg := core.DefaultConfig()

		sessionCfg := model.buildSessionConfig(ctx, messages, cfg)

		if sessionCfg.Model != "gpt-5-mini" {
			t.Errorf("expected model 'gpt-5-mini', got %q", sessionCfg.Model)
		}
		if sessionCfg.ReasoningEffort != "medium" {
			t.Errorf("expected reasoning effort 'medium', got %q", sessionCfg.ReasoningEffort)
		}
		if sessionCfg.InfiniteSessions == nil || *sessionCfg.InfiniteSessions.Enabled {
			t.Error("expected infinite sessions to be disabled")
		}
		if sessionCfg.AvailableTools == nil {
			t.Fatal("expected available tools allowlist to be set")
		}
		if len(sessionCfg.AvailableTools) != 0 {
			t.Errorf("expected no available tools, got %v", sessionCfg.AvailableTools)
		}
		if !stringSliceContains(sessionCfg.ExcludedTools, "web_fetch") {
			t.Fatalf("expected web_fetch to be excluded by default, got %v", sessionCfg.ExcludedTools)
		}
		if !stringSliceContains(sessionCfg.ExcludedTools, "task") {
			t.Fatalf("expected task to be excluded by default, got %v", sessionCfg.ExcludedTools)
		}
	})

	t.Run("with system message", func(t *testing.T) {
		model := &ChatModel{opts: DefaultOptions()}

		messages := []core.Message{
			core.NewSystemMessage("You are helpful"),
			core.NewHumanMessage("hello"),
		}
		cfg := core.DefaultConfig()

		sessionCfg := model.buildSessionConfig(ctx, messages, cfg)

		if sessionCfg.SystemMessage == nil {
			t.Fatal("expected system message config")
		}
		if sessionCfg.SystemMessage.Mode != "replace" {
			t.Errorf("expected mode 'replace', got %q", sessionCfg.SystemMessage.Mode)
		}
		if sessionCfg.SystemMessage.Content != "You are helpful" {
			t.Errorf("expected content 'You are helpful', got %q", sessionCfg.SystemMessage.Content)
		}
	})

	t.Run("with structured output", func(t *testing.T) {
		model := &ChatModel{
			opts: DefaultOptions(),
			structuredSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
				},
			},
		}

		messages := []core.Message{core.NewHumanMessage("hello")}
		cfg := core.DefaultConfig()

		sessionCfg := model.buildSessionConfig(ctx, messages, cfg)

		if sessionCfg.SystemMessage == nil {
			t.Fatal("expected system message config for structured output")
		}
		if sessionCfg.SystemMessage.Mode != "append" {
			t.Errorf("expected mode 'append', got %q", sessionCfg.SystemMessage.Mode)
		}
		if !contains(sessionCfg.SystemMessage.Content, "JSON schema") {
			t.Error("expected system message to contain JSON schema instructions")
		}
	})

	t.Run("with bound tools", func(t *testing.T) {
		model := &ChatModel{
			opts: DefaultOptions(),
			boundTools: []llms.ToolDefinition{
				{Name: "calculator", Description: "Does math", Parameters: map[string]any{"type": "object"}},
			},
		}

		messages := []core.Message{core.NewHumanMessage("hello")}
		cfg := core.DefaultConfig()

		sessionCfg := model.buildSessionConfig(ctx, messages, cfg)

		if len(sessionCfg.Tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(sessionCfg.Tools))
		}
		if sessionCfg.Tools[0].Name != "calculator" {
			t.Errorf("expected tool name 'calculator', got %q", sessionCfg.Tools[0].Name)
		}
		if len(sessionCfg.AvailableTools) != 1 || sessionCfg.AvailableTools[0] != "calculator" {
			t.Errorf("expected available tools [calculator], got %v", sessionCfg.AvailableTools)
		}
	})

	t.Run("with bound skills", func(t *testing.T) {
		model := &ChatModel{
			opts: DefaultOptions(),
			boundSkills: []llms.SkillDefinition{
				{Name: "review", Description: "Reviews changes", Instructions: "Focus on regressions."},
			},
		}

		messages := []core.Message{core.NewHumanMessage("hello")}
		cfg := core.DefaultConfig()

		sessionCfg := model.buildSessionConfig(ctx, messages, cfg)

		if len(sessionCfg.Tools) != 0 {
			t.Fatalf("expected bound skills to remain a no-op for session tools, got %d tools", len(sessionCfg.Tools))
		}
		if sessionCfg.AvailableTools == nil {
			t.Fatal("expected available tools allowlist to be set")
		}
		if len(sessionCfg.AvailableTools) != 0 {
			t.Fatalf("expected no available tools from bound skills, got %v", sessionCfg.AvailableTools)
		}
	})

	t.Run("with explicit bridged tools", func(t *testing.T) {
		model := &ChatModel{
			opts: &Options{
				Model: "gpt-5-mini",
				Tools: []tools.Tool{
					&mockTool{
						name:        "weather",
						description: "Gets weather",
						schema:      map[string]any{"type": "object"},
						runFunc: func(ctx context.Context, input string) (string, error) {
							return "sunny", nil
						},
					},
				},
			},
		}

		messages := []core.Message{core.NewHumanMessage("hello")}
		cfg := core.DefaultConfig()

		sessionCfg := model.buildSessionConfig(ctx, messages, cfg)

		if len(sessionCfg.Tools) != 1 {
			t.Fatalf("expected 1 tool, got %d", len(sessionCfg.Tools))
		}
		if sessionCfg.Tools[0].Name != "weather" {
			t.Errorf("expected tool name 'weather', got %q", sessionCfg.Tools[0].Name)
		}
		if sessionCfg.Tools[0].Handler == nil {
			t.Fatal("expected bridged tool handler to be set")
		}
		if len(sessionCfg.AvailableTools) != 1 || sessionCfg.AvailableTools[0] != "weather" {
			t.Errorf("expected available tools [weather], got %v", sessionCfg.AvailableTools)
		}
	})

	t.Run("deduplicates available tools", func(t *testing.T) {
		model := &ChatModel{
			opts: &Options{
				Model: "gpt-5-mini",
				Tools: []tools.Tool{
					&mockTool{
						name:        "calculator",
						description: "Does math",
						schema:      map[string]any{"type": "object"},
						runFunc: func(ctx context.Context, input string) (string, error) {
							return "42", nil
						},
					},
				},
			},
			boundTools: []llms.ToolDefinition{
				{Name: "calculator", Description: "Does math", Parameters: map[string]any{"type": "object"}},
			},
		}

		messages := []core.Message{core.NewHumanMessage("hello")}
		cfg := core.DefaultConfig()

		sessionCfg := model.buildSessionConfig(ctx, messages, cfg)

		if len(sessionCfg.AvailableTools) != 1 || sessionCfg.AvailableTools[0] != "calculator" {
			t.Errorf("expected deduplicated available tools [calculator], got %v", sessionCfg.AvailableTools)
		}
	})

	t.Run("model override from config", func(t *testing.T) {
		model := &ChatModel{opts: &Options{Model: "gpt-5-mini"}}

		messages := []core.Message{core.NewHumanMessage("hello")}
		cfg := core.DefaultConfig()
		cfg.Configurable = map[string]any{
			llms.ConfigKeyModel: "claude-sonnet-4.5",
		}

		sessionCfg := model.buildSessionConfig(ctx, messages, cfg)

		if sessionCfg.Model != "claude-sonnet-4.5" {
			t.Errorf("expected model 'claude-sonnet-4.5', got %q", sessionCfg.Model)
		}
	})

	t.Run("with permission handler", func(t *testing.T) {
		handler := func(req copilot.PermissionRequest, inv copilot.PermissionInvocation) (copilot.PermissionRequestResult, error) {
			return copilot.PermissionRequestResult{
				Kind: copilot.PermissionRequestResultKindApproved,
			}, nil
		}

		model := &ChatModel{
			opts: &Options{
				Model:               "gpt-5-mini",
				OnPermissionRequest: handler,
			},
		}

		messages := []core.Message{core.NewHumanMessage("hello")}
		cfg := core.DefaultConfig()

		sessionCfg := model.buildSessionConfig(ctx, messages, cfg)

		if sessionCfg.OnPermissionRequest == nil {
			t.Error("expected permission handler to be set in session config")
		}
	})
}

func TestBridgeTools(t *testing.T) {
	ctx := context.Background()

	t.Run("empty tools", func(t *testing.T) {
		sdkTools := bridgeTools(ctx, nil)
		if sdkTools != nil {
			t.Errorf("expected nil for empty tools, got %v", sdkTools)
		}
	})

	t.Run("single tool", func(t *testing.T) {
		tool := &mockTool{
			name:        "calculator",
			description: "Does math",
			schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"expression": map[string]any{"type": "string"},
				},
			},
			runFunc: func(ctx context.Context, input string) (string, error) {
				return "42", nil
			},
		}

		sdkTools := bridgeTools(ctx, []tools.Tool{tool})

		if len(sdkTools) != 1 {
			t.Fatalf("expected 1 SDK tool, got %d", len(sdkTools))
		}

		sdkTool := sdkTools[0]
		if sdkTool.Name != "calculator" {
			t.Errorf("expected name 'calculator', got %q", sdkTool.Name)
		}
		if sdkTool.Description != "Does math" {
			t.Errorf("expected description 'Does math', got %q", sdkTool.Description)
		}
		if sdkTool.Handler == nil {
			t.Fatal("expected handler to be set")
		}

		// Test the handler.
		result, err := sdkTool.Handler(copilot.ToolInvocation{
			Arguments: map[string]any{"expression": "2+2"},
		})
		if err != nil {
			t.Fatalf("handler returned error: %v", err)
		}
		if result.TextResultForLLM != "42" {
			t.Errorf("expected result '42', got %q", result.TextResultForLLM)
		}
		if result.ResultType != "success" {
			t.Errorf("expected result type 'success', got %q", result.ResultType)
		}
	})

	t.Run("tool with string arguments", func(t *testing.T) {
		var receivedInput string
		tool := &mockTool{
			name:        "echo",
			description: "Echoes input",
			schema:      map[string]any{"type": "object"},
			runFunc: func(ctx context.Context, input string) (string, error) {
				receivedInput = input
				return input, nil
			},
		}

		sdkTools := bridgeTools(ctx, []tools.Tool{tool})
		result, err := sdkTools[0].Handler(copilot.ToolInvocation{
			Arguments: "hello world",
		})

		if err != nil {
			t.Fatalf("handler returned error: %v", err)
		}
		if receivedInput != "hello world" {
			t.Errorf("expected input 'hello world', got %q", receivedInput)
		}
		if result.TextResultForLLM != "hello world" {
			t.Errorf("expected result 'hello world', got %q", result.TextResultForLLM)
		}
	})
}

func TestParseResponse(t *testing.T) {
	t.Run("basic response", func(t *testing.T) {
		content := "Hello, world!"
		event := &copilot.SessionEvent{
			Data: copilot.Data{
				Content: &content,
			},
		}

		result := parseResponse(event)

		if len(result.Generations) != 1 {
			t.Fatalf("expected 1 generation, got %d", len(result.Generations))
		}

		msg := result.Generations[0].Message
		if msg.Content != "Hello, world!" {
			t.Errorf("expected content 'Hello, world!', got %q", msg.Content)
		}

		if result.LLMOutput["provider"] != "github-copilot" {
			t.Errorf("expected provider 'github-copilot', got %v", result.LLMOutput["provider"])
		}
	})

	t.Run("with token usage", func(t *testing.T) {
		content := "Hello"
		inputTokens := float64(10)
		outputTokens := float64(5)

		event := &copilot.SessionEvent{
			Data: copilot.Data{
				Content:      &content,
				InputTokens:  &inputTokens,
				OutputTokens: &outputTokens,
			},
		}

		result := parseResponse(event)

		msg := result.Generations[0].Message
		if msg.UsageMetadata == nil {
			t.Fatal("expected usage metadata")
		}
		if msg.UsageMetadata.InputTokens != 10 {
			t.Errorf("expected 10 input tokens, got %d", msg.UsageMetadata.InputTokens)
		}
		if msg.UsageMetadata.OutputTokens != 5 {
			t.Errorf("expected 5 output tokens, got %d", msg.UsageMetadata.OutputTokens)
		}
		if msg.UsageMetadata.TotalTokens != 15 {
			t.Errorf("expected 15 total tokens, got %d", msg.UsageMetadata.TotalTokens)
		}

		usage := result.LLMOutput["token_usage"].(llms.TokenUsage)
		if usage.PromptTokens != 10 {
			t.Errorf("expected 10 prompt tokens, got %d", usage.PromptTokens)
		}
		if usage.CompletionTokens != 5 {
			t.Errorf("expected 5 completion tokens, got %d", usage.CompletionTokens)
		}
	})

	t.Run("nil event", func(t *testing.T) {
		result := parseResponse(nil)

		if len(result.Generations) != 1 {
			t.Fatalf("expected 1 generation, got %d", len(result.Generations))
		}

		msg := result.Generations[0].Message
		if msg.Content != "" {
			t.Errorf("expected empty content, got %q", msg.Content)
		}
	})
}

func TestDefaultOptions(t *testing.T) {
	opts := DefaultOptions()

	if opts.Model != "gpt-5-mini" {
		t.Errorf("expected default model 'gpt-5-mini', got %q", opts.Model)
	}
	if opts.LogLevel != "error" {
		t.Errorf("expected default log level 'error', got %q", opts.LogLevel)
	}
	if opts.MaxConcurrency != 5 {
		t.Errorf("expected default max concurrency 5, got %d", opts.MaxConcurrency)
	}
}

func TestOptionFuncs(t *testing.T) {
	t.Run("WithGithubToken", func(t *testing.T) {
		opts := DefaultOptions()
		WithGithubToken("test-token")(opts)
		if opts.GithubToken != "test-token" {
			t.Errorf("expected token 'test-token', got %q", opts.GithubToken)
		}
	})

	t.Run("WithModelName", func(t *testing.T) {
		opts := DefaultOptions()
		WithModelName("gpt-5")(opts)
		if opts.Model != "gpt-5" {
			t.Errorf("expected model 'gpt-5', got %q", opts.Model)
		}
	})

	t.Run("WithCLIPath", func(t *testing.T) {
		opts := DefaultOptions()
		WithCLIPath("/custom/path")(opts)
		if opts.CLIPath != "/custom/path" {
			t.Errorf("expected CLI path '/custom/path', got %q", opts.CLIPath)
		}
	})

	t.Run("WithLogLevel", func(t *testing.T) {
		opts := DefaultOptions()
		WithLogLevel("debug")(opts)
		if opts.LogLevel != "debug" {
			t.Errorf("expected log level 'debug', got %q", opts.LogLevel)
		}
	})

	t.Run("WithMaxConcurrency", func(t *testing.T) {
		opts := DefaultOptions()
		WithMaxConcurrency(10)(opts)
		if opts.MaxConcurrency != 10 {
			t.Errorf("expected max concurrency 10, got %d", opts.MaxConcurrency)
		}
	})

	t.Run("WithReasoningEffort", func(t *testing.T) {
		opts := DefaultOptions()
		WithReasoningEffort("high")(opts)
		if opts.ReasoningEffort != "high" {
			t.Errorf("expected reasoning effort 'high', got %q", opts.ReasoningEffort)
		}
	})

	t.Run("WithTools", func(t *testing.T) {
		opts := DefaultOptions()
		tool := &mockTool{name: "test"}
		WithTools(tool)(opts)
		if len(opts.Tools) != 1 {
			t.Errorf("expected 1 tool, got %d", len(opts.Tools))
		}
		if opts.Tools[0].Name() != "test" {
			t.Errorf("expected tool name 'test', got %q", opts.Tools[0].Name())
		}
	})

	t.Run("WithPermissionHandler", func(t *testing.T) {
		opts := DefaultOptions()
		handler := func(req copilot.PermissionRequest, inv copilot.PermissionInvocation) (copilot.PermissionRequestResult, error) {
			return copilot.PermissionRequestResult{
				Kind: copilot.PermissionRequestResultKindApproved,
			}, nil
		}
		WithPermissionHandler(handler)(opts)
		if opts.OnPermissionRequest == nil {
			t.Error("expected permission handler to be set")
		}
	})
}

func TestChatModelImplementsInterface(t *testing.T) {
	var _ llms.ChatModel = (*ChatModel)(nil)
}

// Helper function for string contains check.
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func stringSliceContains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
