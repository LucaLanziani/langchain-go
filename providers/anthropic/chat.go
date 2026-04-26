package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
)

const anthropicAPIVersion = "2023-06-01"

// ChatModel is the Anthropic Messages API implementation.
type ChatModel struct {
	opts             *Options
	client           *http.Client
	boundTools       []llms.ToolDefinition
	boundSkills      []llms.SkillDefinition
	structuredSchema map[string]any
	name             string
}

// New creates a new Anthropic ChatModel.
func New(optFns ...OptionFunc) *ChatModel {
	opts := DefaultOptions()
	for _, fn := range optFns {
		fn(opts)
	}
	if opts.APIKey == "" {
		opts.APIKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	return &ChatModel{
		opts:   opts,
		client: &http.Client{},
	}
}

// GetName returns the name of this model.
func (m *ChatModel) GetName() string {
	if m.name != "" {
		return m.name
	}
	return "ChatAnthropic"
}

// BindTools returns a copy of the model with tools bound.
func (m *ChatModel) BindTools(tools ...llms.ToolDefinition) llms.ChatModel {
	cp := *m
	cp.boundTools = append(append([]llms.ToolDefinition(nil), m.boundTools...), tools...)
	return &cp
}

// BindSkills returns a copy of the model with skills bound.
func (m *ChatModel) BindSkills(skills ...llms.SkillDefinition) llms.ChatModel {
	cp := *m
	cp.boundSkills = append(append([]llms.SkillDefinition(nil), m.boundSkills...), skills...)
	return &cp
}

// WithStructuredOutput returns a copy configured for structured output.
func (m *ChatModel) WithStructuredOutput(schema map[string]any) llms.ChatModel {
	cp := *m
	cp.structuredSchema = core.CloneMap(schema)
	return &cp
}

// Invoke sends messages to Anthropic and returns the AI response.
func (m *ChatModel) Invoke(ctx context.Context, input []core.Message, opts ...core.Option) (*core.AIMessage, error) {
	result, err := m.Generate(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	if len(result.Generations) == 0 {
		return nil, fmt.Errorf("no generations returned")
	}
	return result.Generations[0].Message, nil
}

// Generate performs a chat completion with full result details.
func (m *ChatModel) Generate(ctx context.Context, messages []core.Message, opts ...core.Option) (*llms.ChatResult, error) {
	cfg := core.ApplyOptions(opts...)
	reqBody, err := m.buildRequest(messages, cfg, false)
	if err != nil {
		return nil, err
	}

	respBody, err := m.doRequest(ctx, "/messages", reqBody)
	if err != nil {
		return nil, err
	}

	return m.parseResponse(respBody)
}

// Stream sends messages and streams the response.
func (m *ChatModel) Stream(ctx context.Context, input []core.Message, opts ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
	cfg := core.ApplyOptions(opts...)
	reqBody, err := m.buildRequest(input, cfg, true)
	if err != nil {
		return nil, err
	}

	reqJSON, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.opts.BaseURL+"/messages", bytes.NewReader(reqJSON))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	m.setHeaders(req)

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("Anthropic API error (status %d): %s", resp.StatusCode, string(body))
	}

	ch := make(chan core.StreamChunk[*core.AIMessage], 64)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		m.streamResponse(resp.Body, ch)
	}()

	return core.NewStreamIterator(ch), nil
}

// Batch performs multiple chat completions.
func (m *ChatModel) Batch(ctx context.Context, inputs [][]core.Message, opts ...core.Option) ([]*core.AIMessage, error) {
	return core.Batch(ctx, inputs, opts, m.Invoke)
}

// buildRequest constructs the Anthropic API request body.
func (m *ChatModel) buildRequest(messages []core.Message, cfg *core.RunnableConfig, stream bool) (map[string]any, error) {
	model := m.opts.Model
	if v, ok := cfg.Configurable[llms.ConfigKeyModel]; ok {
		if s, ok := v.(string); ok {
			model = s
		}
	}

	// Anthropic requires system message to be separate.
	var system string
	var apiMessages []map[string]any
	for _, msg := range messages {
		if msg.GetType() == core.MessageTypeSystem {
			system = msg.GetContent()
			continue
		}
		apiMessages = append(apiMessages, m.messageToAPI(msg))
	}

	maxTokens := m.opts.MaxTokens
	if mt, ok := cfg.Configurable[llms.ConfigKeyMaxTokens]; ok {
		if i, ok := mt.(int); ok {
			maxTokens = i
		}
	}

	req := map[string]any{
		"model":      model,
		"messages":   apiMessages,
		"max_tokens": maxTokens,
	}

	if system != "" {
		req["system"] = system
	}

	if stream {
		req["stream"] = true
	}

	// Temperature
	if temp, ok := cfg.Configurable[llms.ConfigKeyTemperature]; ok {
		req["temperature"] = temp
	} else if m.opts.Temperature != nil {
		req["temperature"] = *m.opts.Temperature
	}

	// TopP
	if tp, ok := cfg.Configurable[llms.ConfigKeyTopP]; ok {
		req["top_p"] = tp
	} else if m.opts.TopP != nil {
		req["top_p"] = *m.opts.TopP
	}

	// Stop
	stop := cfg.Stop
	if len(stop) == 0 {
		stop = m.opts.Stop
	}
	if len(stop) > 0 {
		req["stop_sequences"] = stop
	}

	// Extended Thinking: when enabled, temperature must be 1 and top_p/top_k
	// must not be set (Anthropic API requirement).
	if m.opts.ThinkingBudget > 0 {
		if err := validateThinkingConfig(cfg, m.opts); err != nil {
			return nil, err
		}
		req["thinking"] = map[string]any{
			"type":          "enabled",
			"budget_tokens": m.opts.ThinkingBudget,
		}
		req["temperature"] = 1 // forced; overrides any Temperature option
		delete(req, "top_p")
		delete(req, "top_k")
	}

	// Tools
	if len(m.boundTools) > 0 {
		tools := make([]map[string]any, len(m.boundTools))
		for i, t := range m.boundTools {
			tools[i] = map[string]any{
				"name":         t.Name,
				"description":  t.Description,
				"input_schema": t.Parameters,
			}
		}
		req["tools"] = tools
	}

	if skills := skillsToRequest(m.boundSkills); len(skills) > 0 {
		req["skills"] = skills
	}

	return req, nil
}

func skillsToRequest(skills []llms.SkillDefinition) []map[string]any {
	_ = skills
	return nil
}

// messageToAPI converts a core.Message to the Anthropic API format.
func (m *ChatModel) messageToAPI(msg core.Message) map[string]any {
	switch msg.GetType() {
	case core.MessageTypeHuman:
		return map[string]any{
			"role":    "user",
			"content": msg.GetContent(),
		}
	case core.MessageTypeAI:
		apiMsg := map[string]any{
			"role": "assistant",
		}
		if ai, ok := msg.(*core.AIMessage); ok && len(ai.ToolCalls) > 0 {
			content := []map[string]any{}
			if ai.Content != "" {
				content = append(content, map[string]any{
					"type": "text",
					"text": ai.Content,
				})
			}
			for _, tc := range ai.ToolCalls {
				var input any
				_ = json.Unmarshal(tc.Args, &input)
				content = append(content, map[string]any{
					"type":  "tool_use",
					"id":    tc.ID,
					"name":  tc.Name,
					"input": input,
				})
			}
			apiMsg["content"] = content
		} else {
			apiMsg["content"] = msg.GetContent()
		}
		return apiMsg
	case core.MessageTypeTool:
		tm := msg.(*core.ToolMessage)
		return map[string]any{
			"role": "user",
			"content": []map[string]any{
				{
					"type":        "tool_result",
					"tool_use_id": tm.ToolCallID,
					"content":     tm.Content,
				},
			},
		}
	default:
		return map[string]any{
			"role":    "user",
			"content": msg.GetContent(),
		}
	}
}

// doRequest sends an HTTP request and returns the response body.
func (m *ChatModel) doRequest(ctx context.Context, path string, body any) ([]byte, error) {
	reqJSON, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.opts.BaseURL+path, bytes.NewReader(reqJSON))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	m.setHeaders(req)

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Anthropic API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

// setHeaders sets the standard headers for Anthropic API requests.
func (m *ChatModel) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", m.opts.APIKey)
	req.Header.Set("anthropic-version", anthropicAPIVersion)
	if m.opts.ThinkingBudget > 0 {
		// Required beta for extended thinking (streaming interleaved thinking blocks).
		req.Header.Set("anthropic-beta", "interleaved-thinking-2025-05-14")
	}
}

// parseResponse parses the Anthropic messages API response.
func (m *ChatModel) parseResponse(body []byte) (*llms.ChatResult, error) {
	var resp anthropicResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	aiMsg := m.responseToMessage(&resp)

	result := &llms.ChatResult{
		Generations: []*llms.ChatGeneration{
			{
				Message: aiMsg,
				GenerationInfo: map[string]any{
					"stop_reason": resp.StopReason,
				},
			},
		},
		LLMOutput: map[string]any{
			"model": resp.Model,
		},
	}

	if resp.Usage != nil {
		result.LLMOutput["token_usage"] = llms.TokenUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		}
		aiMsg.UsageMetadata = &core.UsageMetadata{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
			TotalTokens:  resp.Usage.InputTokens + resp.Usage.OutputTokens,
		}
	}

	return result, nil
}

// responseToMessage converts an Anthropic response to a core.AIMessage.
func (m *ChatModel) responseToMessage(resp *anthropicResponse) *core.AIMessage {
	var content strings.Builder
	var thinking strings.Builder
	var toolCalls []core.ToolCall

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			content.WriteString(block.Text)
		case "thinking":
			thinking.WriteString(block.Text)
		case "tool_use":
			argsJSON, _ := json.Marshal(block.Input)
			toolCalls = append(toolCalls, core.ToolCall{
				ID:   block.ID,
				Name: block.Name,
				Args: argsJSON,
				Type: "function",
			})
		}
	}

	var message *core.AIMessage
	if len(toolCalls) > 0 {
		message = core.NewAIMessageWithToolCalls(content.String(), toolCalls)
	} else {
		message = core.NewAIMessage(content.String())
	}
	if thinking.Len() > 0 {
		if message.AdditionalKwargs == nil {
			message.AdditionalKwargs = make(map[string]any)
		}
		message.AdditionalKwargs["thinking"] = thinking.String()
	}
	return message
}

func validateThinkingConfig(cfg *core.RunnableConfig, opts *Options) error {
	if temp, ok := cfg.Configurable[llms.ConfigKeyTemperature]; ok {
		if value, ok := temp.(float64); ok && value != 1 {
			return fmt.Errorf("anthropic: thinking budget requires temperature=1")
		}
	}
	if opts.Temperature != nil && *opts.Temperature != 1 {
		return fmt.Errorf("anthropic: thinking budget requires temperature=1")
	}
	if _, ok := cfg.Configurable[llms.ConfigKeyTopP]; ok {
		return fmt.Errorf("anthropic: thinking budget is incompatible with top_p")
	}
	if opts.TopP != nil {
		return fmt.Errorf("anthropic: thinking budget is incompatible with top_p")
	}
	return nil
}

// streamResponse reads SSE events from the Anthropic streaming response.
func (m *ChatModel) streamResponse(body io.Reader, ch chan<- core.StreamChunk[*core.AIMessage]) {
	scanner := bufio.NewScanner(body)
	var contentBuilder strings.Builder
	var currentToolCall *toolCallAccumulator
	var toolCalls []core.ToolCall

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		var event anthropicStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		switch event.Type {
		case "content_block_start":
			if event.ContentBlock != nil && event.ContentBlock.Type == "tool_use" {
				currentToolCall = &toolCallAccumulator{
					id:   event.ContentBlock.ID,
					name: event.ContentBlock.Name,
				}
			}

		case "content_block_delta":
			if event.Delta != nil {
				switch event.Delta.Type {
				case "text_delta":
					contentBuilder.WriteString(event.Delta.Text)
					msg := core.NewAIMessage(event.Delta.Text)
					ch <- core.StreamChunk[*core.AIMessage]{Value: msg}

				case "thinking_delta":
					// Emit the thinking chunk with a metadata flag so callers
					// can distinguish it from the main response text.
					msg := core.NewAIMessage("")
					if msg.AdditionalKwargs == nil {
						msg.AdditionalKwargs = make(map[string]any)
					}
					msg.AdditionalKwargs["thinking"] = event.Delta.Thinking
					ch <- core.StreamChunk[*core.AIMessage]{Value: msg}

				case "input_json_delta":
					if currentToolCall != nil {
						currentToolCall.args += event.Delta.PartialJSON
					}
				}
			}

		case "content_block_stop":
			if currentToolCall != nil {
				toolCalls = append(toolCalls, core.ToolCall{
					ID:   currentToolCall.id,
					Name: currentToolCall.name,
					Args: json.RawMessage(currentToolCall.args),
					Type: "function",
				})
				currentToolCall = nil
			}

		case "message_stop":
			if len(toolCalls) > 0 {
				msg := core.NewAIMessageWithToolCalls(contentBuilder.String(), toolCalls)
				ch <- core.StreamChunk[*core.AIMessage]{Value: msg}
			}
		}
	}
}

type toolCallAccumulator struct {
	id   string
	name string
	args string
}

// Anthropic API types

type anthropicResponse struct {
	ID         string             `json:"id"`
	Type       string             `json:"type"`
	Role       string             `json:"role"`
	Content    []anthropicContent `json:"content"`
	Model      string             `json:"model"`
	StopReason string             `json:"stop_reason"`
	Usage      *anthropicUsage    `json:"usage,omitempty"`
}

type anthropicContent struct {
	Type  string `json:"type"`
	Text  string `json:"text,omitempty"`
	ID    string `json:"id,omitempty"`
	Name  string `json:"name,omitempty"`
	Input any    `json:"input,omitempty"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type anthropicStreamEvent struct {
	Type         string            `json:"type"`
	ContentBlock *anthropicContent `json:"content_block,omitempty"`
	Delta        *anthropicDelta   `json:"delta,omitempty"`
	Index        int               `json:"index,omitempty"`
}

type anthropicDelta struct {
	Type        string `json:"type,omitempty"`
	Text        string `json:"text,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
}

// Ensure ChatModel implements llms.ChatModel.
var _ llms.ChatModel = (*ChatModel)(nil)
