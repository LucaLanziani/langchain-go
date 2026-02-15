package copilot

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"

	copilot "github.com/github/copilot-sdk/go"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
	"github.com/LucaLanziani/langchain-go/tools"
)

// Ensure ChatModel implements llms.ChatModel.
var _ llms.ChatModel = (*ChatModel)(nil)

// ChatModel is the GitHub Copilot chat model implementation backed by the Copilot SDK.
type ChatModel struct {
	opts             *Options
	client           *copilot.Client
	boundTools       []llms.ToolDefinition
	structuredSchema map[string]any
	name             string
}

// New creates a new GitHub Copilot ChatModel.
// The returned model must be closed with Close() when no longer needed.
func New(ctx context.Context, optFns ...OptionFunc) (*ChatModel, error) {
	opts := DefaultOptions()
	for _, fn := range optFns {
		fn(opts)
	}
	if opts.GithubToken == "" {
		opts.GithubToken = os.Getenv("GITHUB_TOKEN")
	}
	if opts.GithubToken == "" {
		if out, err := exec.Command("gh", "auth", "token").Output(); err == nil {
			opts.GithubToken = strings.TrimSpace(string(out))
		}
	}

	clientOpts := &copilot.ClientOptions{
		LogLevel: opts.LogLevel,
	}
	if opts.GithubToken != "" {
		clientOpts.GithubToken = opts.GithubToken
	}
	if opts.CLIPath != "" {
		clientOpts.CLIPath = opts.CLIPath
	}

	client := copilot.NewClient(clientOpts)
	if err := client.Start(ctx); err != nil {
		return nil, fmt.Errorf("copilot: failed to start client: %w", err)
	}

	return &ChatModel{
		opts:   opts,
		client: client,
	}, nil
}

// Close stops the Copilot CLI server and releases resources.
func (m *ChatModel) Close() error {
	if m.client != nil {
		return m.client.Stop()
	}
	return nil
}

// GetName returns the name of this model.
func (m *ChatModel) GetName() string {
	if m.name != "" {
		return m.name
	}
	return "ChatGitHubCopilot"
}

// BindTools returns a copy of the model with tools bound.
func (m *ChatModel) BindTools(toolDefs ...llms.ToolDefinition) llms.ChatModel {
	cp := *m
	cp.boundTools = append(cp.boundTools, toolDefs...)
	return &cp
}

// WithStructuredOutput returns a copy of the model configured for structured output.
func (m *ChatModel) WithStructuredOutput(schema map[string]any) llms.ChatModel {
	cp := *m
	cp.structuredSchema = schema
	return &cp
}

// Invoke sends messages to the Copilot API and returns the AI response.
func (m *ChatModel) Invoke(ctx context.Context, input []core.Message, opts ...core.Option) (*core.AIMessage, error) {
	result, err := m.Generate(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	if len(result.Generations) == 0 {
		return nil, fmt.Errorf("copilot: no generations returned")
	}
	return result.Generations[0].Message, nil
}

// Generate performs a chat completion with full result details.
func (m *ChatModel) Generate(ctx context.Context, messages []core.Message, opts ...core.Option) (*llms.ChatResult, error) {
	cfg := core.ApplyOptions(opts...)

	sessionCfg := m.buildSessionConfig(messages, cfg)
	session, err := m.client.CreateSession(ctx, sessionCfg)
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to create session: %w", err)
	}
	defer session.Destroy()

	prompt := messagesToPrompt(messages)
	response, err := session.SendAndWait(ctx, copilot.MessageOptions{
		Prompt: prompt,
	})
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to send message: %w", err)
	}

	return parseResponse(response), nil
}

// Stream sends messages and streams the response token by token.
func (m *ChatModel) Stream(ctx context.Context, input []core.Message, opts ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
	cfg := core.ApplyOptions(opts...)

	sessionCfg := m.buildSessionConfig(input, cfg)
	sessionCfg.Streaming = true

	session, err := m.client.CreateSession(ctx, sessionCfg)
	if err != nil {
		return nil, fmt.Errorf("copilot: failed to create session: %w", err)
	}

	ch := make(chan core.StreamChunk[*core.AIMessage], 64)

	// Track whether the session has finished so we can clean up.
	done := make(chan struct{})

	session.On(func(event copilot.SessionEvent) {
		switch event.Type {
		case copilot.AssistantMessageDelta:
			if event.Data.DeltaContent != nil {
				msg := core.NewAIMessage(*event.Data.DeltaContent)
				ch <- core.StreamChunk[*core.AIMessage]{Value: msg}
			}

		case copilot.AssistantMessage:
			// Don't repeat content — deltas already delivered it token by token.
			msg := core.NewAIMessage("")
			if event.Data.InputTokens != nil || event.Data.OutputTokens != nil {
				inputTokens := 0
				outputTokens := 0
				if event.Data.InputTokens != nil {
					inputTokens = int(*event.Data.InputTokens)
				}
				if event.Data.OutputTokens != nil {
					outputTokens = int(*event.Data.OutputTokens)
				}
				msg.UsageMetadata = &core.UsageMetadata{
					InputTokens:  inputTokens,
					OutputTokens: outputTokens,
					TotalTokens:  inputTokens + outputTokens,
				}
			}
			ch <- core.StreamChunk[*core.AIMessage]{Value: msg}

		case copilot.SessionError:
			errMsg := "unknown error"
			if event.Data.Message != nil {
				errMsg = *event.Data.Message
			}
			ch <- core.StreamChunk[*core.AIMessage]{
				Err: fmt.Errorf("copilot: session error: %s", errMsg),
			}

		case copilot.SessionIdle:
			close(done)
		}
	})

	prompt := messagesToPrompt(input)
	if _, err := session.Send(ctx, copilot.MessageOptions{
		Prompt: prompt,
	}); err != nil {
		session.Destroy()
		close(ch)
		return nil, fmt.Errorf("copilot: failed to send message: %w", err)
	}

	// Clean up session when streaming is complete.
	go func() {
		defer close(ch)
		select {
		case <-done:
		case <-ctx.Done():
		}
		session.Destroy()
	}()

	return core.NewStreamIterator(ch), nil
}

// Batch performs multiple chat completions in parallel.
// Concurrency is controlled by the MaxConcurrency option (default 5).
func (m *ChatModel) Batch(ctx context.Context, inputs [][]core.Message, opts ...core.Option) ([]*core.AIMessage, error) {
	results := make([]*core.AIMessage, len(inputs))
	errs := make([]error, len(inputs))

	maxConc := m.opts.MaxConcurrency
	if maxConc <= 0 {
		maxConc = 5
	}
	sem := make(chan struct{}, maxConc)

	var wg sync.WaitGroup
	for i, input := range inputs {
		wg.Add(1)
		go func(idx int, msgs []core.Message) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			result, err := m.Invoke(ctx, msgs, opts...)
			if err != nil {
				errs[idx] = fmt.Errorf("batch item %d: %w", idx, err)
				return
			}
			results[idx] = result
		}(i, input)
	}

	wg.Wait()

	// Return first error encountered.
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}

	return results, nil
}

// buildSessionConfig constructs the SDK SessionConfig from options and runtime config.
func (m *ChatModel) buildSessionConfig(messages []core.Message, cfg *core.RunnableConfig) *copilot.SessionConfig {
	model := m.opts.Model
	if v, ok := cfg.Configurable[llms.ConfigKeyModel]; ok {
		model = v.(string)
	}

	sessionCfg := &copilot.SessionConfig{
		Model: model,
		// Disable infinite sessions — each request is atomic.
		InfiniteSessions: &copilot.InfiniteSessionConfig{
			Enabled: copilot.Bool(false),
		},
	}

	// Extract system message and configure it.
	if sysMsg := extractSystemMessage(messages); sysMsg != "" {
		systemContent := sysMsg
		// If structured output is requested, append schema instructions to the system message.
		if m.structuredSchema != nil {
			schemaJSON, err := json.Marshal(m.structuredSchema)
			if err == nil {
				systemContent += fmt.Sprintf(
					"\n\nYou must respond with valid JSON that conforms to this JSON schema:\n%s",
					string(schemaJSON),
				)
			}
		}
		sessionCfg.SystemMessage = &copilot.SystemMessageConfig{
			Mode:    "replace",
			Content: systemContent,
		}
	} else if m.structuredSchema != nil {
		schemaJSON, err := json.Marshal(m.structuredSchema)
		if err == nil {
			sessionCfg.SystemMessage = &copilot.SystemMessageConfig{
				Mode: "append",
				Content: fmt.Sprintf(
					"\n\nYou must respond with valid JSON that conforms to this JSON schema:\n%s",
					string(schemaJSON),
				),
			}
		}
	}

	// Bridge langchain tools to SDK tools.
	sdkTools := bridgeTools(m.opts.Tools)

	// Also add any bound tool definitions as SDK tools (without handlers, so the SDK
	// will report them but won't auto-execute).
	for _, td := range m.boundTools {
		sdkTools = append(sdkTools, copilot.Tool{
			Name:        td.Name,
			Description: td.Description,
			Parameters:  td.Parameters,
		})
	}

	if len(sdkTools) > 0 {
		sessionCfg.Tools = sdkTools
	}

	return sessionCfg
}

// bridgeTools converts langchain Tool implementations to copilot.Tool structs
// with real handlers, so the SDK manages the tool-calling loop internally.
func bridgeTools(langchainTools []tools.Tool) []copilot.Tool {
	if len(langchainTools) == 0 {
		return nil
	}

	sdkTools := make([]copilot.Tool, len(langchainTools))
	for i, t := range langchainTools {
		tool := t // capture loop variable
		sdkTools[i] = copilot.Tool{
			Name:        tool.Name(),
			Description: tool.Description(),
			Parameters:  tool.ArgsSchema(),
			Handler: func(inv copilot.ToolInvocation) (copilot.ToolResult, error) {
				// Serialize arguments to JSON string for the langchain Tool.Run interface.
				var argsStr string
				switch v := inv.Arguments.(type) {
				case string:
					argsStr = v
				default:
					b, err := json.Marshal(v)
					if err != nil {
						return copilot.ToolResult{
							ResultType: "error",
							Error:      fmt.Sprintf("failed to marshal tool args: %v", err),
						}, nil
					}
					argsStr = string(b)
				}

				result, err := tool.Run(context.Background(), argsStr)
				if err != nil {
					return copilot.ToolResult{
						TextResultForLLM: fmt.Sprintf("Error: %v", err),
						ResultType:       "error",
						Error:            err.Error(),
					}, nil
				}

				return copilot.ToolResult{
					TextResultForLLM: result,
					ResultType:       "success",
				}, nil
			},
		}
	}

	return sdkTools
}

// extractSystemMessage finds the first system message content.
func extractSystemMessage(messages []core.Message) string {
	for _, msg := range messages {
		if msg.GetType() == core.MessageTypeSystem {
			return msg.GetContent()
		}
	}
	return ""
}

// messagesToPrompt converts the conversation messages into a single prompt string,
// skipping system messages (handled separately via SessionConfig.SystemMessage).
// This keeps most of the conversation in a single request to the SDK.
func messagesToPrompt(messages []core.Message) string {
	var parts []string

	for _, msg := range messages {
		switch msg.GetType() {
		case core.MessageTypeSystem:
			// Handled separately via session config.
			continue
		case core.MessageTypeHuman:
			parts = append(parts, msg.GetContent())
		case core.MessageTypeAI:
			parts = append(parts, "Assistant: "+msg.GetContent())
			if ai, ok := msg.(*core.AIMessage); ok && len(ai.ToolCalls) > 0 {
				for _, tc := range ai.ToolCalls {
					parts = append(parts, fmt.Sprintf("[Tool Call: %s(%s)]", tc.Name, string(tc.Args)))
				}
			}
		case core.MessageTypeTool:
			if tm, ok := msg.(*core.ToolMessage); ok {
				parts = append(parts, fmt.Sprintf("[Tool Result (%s): %s]", tm.ToolCallID, msg.GetContent()))
			} else {
				parts = append(parts, "Tool: "+msg.GetContent())
			}
		case core.MessageTypeFunction:
			parts = append(parts, fmt.Sprintf("Function (%s): %s", msg.GetName(), msg.GetContent()))
		default:
			parts = append(parts, msg.GetContent())
		}
	}

	return strings.Join(parts, "\n")
}

// parseResponse converts a SessionEvent into a ChatResult.
func parseResponse(event *copilot.SessionEvent) *llms.ChatResult {
	content := ""
	if event != nil && event.Data.Content != nil {
		content = *event.Data.Content
	}

	aiMsg := core.NewAIMessage(content)

	// Extract usage metadata if available.
	if event != nil {
		inputTokens := 0
		outputTokens := 0
		if event.Data.InputTokens != nil {
			inputTokens = int(*event.Data.InputTokens)
		}
		if event.Data.OutputTokens != nil {
			outputTokens = int(*event.Data.OutputTokens)
		}
		if inputTokens > 0 || outputTokens > 0 {
			aiMsg.UsageMetadata = &core.UsageMetadata{
				InputTokens:  inputTokens,
				OutputTokens: outputTokens,
				TotalTokens:  inputTokens + outputTokens,
			}
		}
	}

	result := &llms.ChatResult{
		LLMOutput: map[string]any{
			"provider": "github-copilot",
		},
	}

	if aiMsg.UsageMetadata != nil {
		result.LLMOutput["token_usage"] = llms.TokenUsage{
			PromptTokens:     aiMsg.UsageMetadata.InputTokens,
			CompletionTokens: aiMsg.UsageMetadata.OutputTokens,
			TotalTokens:      aiMsg.UsageMetadata.TotalTokens,
		}
	}

	result.Generations = []*llms.ChatGeneration{
		{
			Message: aiMsg,
			GenerationInfo: map[string]any{
				"provider": "github-copilot",
			},
		},
	}

	return result
}
