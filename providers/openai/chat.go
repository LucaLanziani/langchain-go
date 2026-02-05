package openai

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

// ChatModel is the OpenAI chat completion implementation.
type ChatModel struct {
	opts             *Options
	client           *http.Client
	boundTools       []llms.ToolDefinition
	structuredSchema map[string]any
	name             string
}

// New creates a new OpenAI ChatModel.
func New(optFns ...OptionFunc) *ChatModel {
	opts := DefaultOptions()
	for _, fn := range optFns {
		fn(opts)
	}
	if opts.APIKey == "" {
		opts.APIKey = os.Getenv("OPENAI_API_KEY")
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
	return "ChatOpenAI"
}

// BindTools returns a copy of the model with tools bound.
func (m *ChatModel) BindTools(tools ...llms.ToolDefinition) llms.ChatModel {
	cp := *m
	cp.boundTools = append(cp.boundTools, tools...)
	return &cp
}

// WithStructuredOutput returns a copy of the model configured for structured output.
func (m *ChatModel) WithStructuredOutput(schema map[string]any) llms.ChatModel {
	cp := *m
	cp.structuredSchema = schema
	return &cp
}

// Invoke sends messages to OpenAI and returns the AI response.
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
	reqBody := m.buildRequest(messages, cfg, false)

	respBody, err := m.doRequest(ctx, "/chat/completions", reqBody)
	if err != nil {
		return nil, err
	}

	return m.parseResponse(respBody)
}

// Stream sends messages and streams the response token by token.
func (m *ChatModel) Stream(ctx context.Context, input []core.Message, opts ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
	cfg := core.ApplyOptions(opts...)
	reqBody := m.buildRequest(input, cfg, true)

	reqJSON, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.opts.BaseURL+"/chat/completions", bytes.NewReader(reqJSON))
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
		return nil, fmt.Errorf("OpenAI API error (status %d): %s", resp.StatusCode, string(body))
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
	results := make([]*core.AIMessage, len(inputs))
	for i, input := range inputs {
		result, err := m.Invoke(ctx, input, opts...)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// buildRequest constructs the OpenAI API request body.
func (m *ChatModel) buildRequest(messages []core.Message, cfg *core.RunnableConfig, stream bool) map[string]any {
	model := m.opts.Model
	if v, ok := cfg.Configurable[llms.ConfigKeyModel]; ok {
		model = v.(string)
	}

	apiMessages := make([]map[string]any, 0, len(messages))
	for _, msg := range messages {
		apiMsg := m.messageToAPI(msg)
		apiMessages = append(apiMessages, apiMsg)
	}

	req := map[string]any{
		"model":    model,
		"messages": apiMessages,
	}

	if stream {
		req["stream"] = true
		req["stream_options"] = map[string]any{"include_usage": true}
	}

	// Temperature
	if temp, ok := cfg.Configurable[llms.ConfigKeyTemperature]; ok {
		req["temperature"] = temp
	} else if m.opts.Temperature != nil {
		req["temperature"] = *m.opts.Temperature
	}

	// MaxTokens
	if mt, ok := cfg.Configurable[llms.ConfigKeyMaxTokens]; ok {
		req["max_tokens"] = mt
	} else if m.opts.MaxTokens != nil {
		req["max_tokens"] = *m.opts.MaxTokens
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
		req["stop"] = stop
	}

	// Tools
	if len(m.boundTools) > 0 {
		tools := make([]map[string]any, len(m.boundTools))
		for i, t := range m.boundTools {
			tools[i] = map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        t.Name,
					"description": t.Description,
					"parameters":  t.Parameters,
				},
			}
		}
		req["tools"] = tools
	}

	// Structured output
	if m.structuredSchema != nil {
		req["response_format"] = map[string]any{
			"type":        "json_schema",
			"json_schema": m.structuredSchema,
		}
	} else if m.opts.ResponseFormat == "json_object" {
		req["response_format"] = map[string]any{"type": "json_object"}
	}

	return req
}

// messageToAPI converts a core.Message to the OpenAI API format.
func (m *ChatModel) messageToAPI(msg core.Message) map[string]any {
	apiMsg := map[string]any{
		"content": msg.GetContent(),
	}

	switch msg.GetType() {
	case core.MessageTypeHuman:
		apiMsg["role"] = "user"
	case core.MessageTypeAI:
		apiMsg["role"] = "assistant"
		if ai, ok := msg.(*core.AIMessage); ok && len(ai.ToolCalls) > 0 {
			toolCalls := make([]map[string]any, len(ai.ToolCalls))
			for i, tc := range ai.ToolCalls {
				toolCalls[i] = map[string]any{
					"id":   tc.ID,
					"type": "function",
					"function": map[string]any{
						"name":      tc.Name,
						"arguments": string(tc.Args),
					},
				}
			}
			apiMsg["tool_calls"] = toolCalls
		}
	case core.MessageTypeSystem:
		apiMsg["role"] = "system"
	case core.MessageTypeTool:
		apiMsg["role"] = "tool"
		if tm, ok := msg.(*core.ToolMessage); ok {
			apiMsg["tool_call_id"] = tm.ToolCallID
		}
	case core.MessageTypeFunction:
		apiMsg["role"] = "function"
		if msg.GetName() != "" {
			apiMsg["name"] = msg.GetName()
		}
	default:
		apiMsg["role"] = "user"
	}

	if msg.GetName() != "" && msg.GetType() != core.MessageTypeFunction {
		apiMsg["name"] = msg.GetName()
	}

	return apiMsg
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
		return nil, fmt.Errorf("OpenAI API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

// setHeaders sets the standard headers for OpenAI API requests.
func (m *ChatModel) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.opts.APIKey)
	if m.opts.Organization != "" {
		req.Header.Set("OpenAI-Organization", m.opts.Organization)
	}
}

// parseResponse parses the OpenAI chat completion response.
func (m *ChatModel) parseResponse(body []byte) (*llms.ChatResult, error) {
	var resp openAIChatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	result := &llms.ChatResult{
		LLMOutput: map[string]any{
			"model": resp.Model,
		},
	}

	if resp.Usage != nil {
		result.LLMOutput["token_usage"] = llms.TokenUsage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
	}

	for _, choice := range resp.Choices {
		aiMsg := core.NewAIMessage(choice.Message.Content)
		aiMsg.ResponseMetadata = map[string]any{
			"finish_reason": choice.FinishReason,
		}

		if len(choice.Message.ToolCalls) > 0 {
			toolCalls := make([]core.ToolCall, len(choice.Message.ToolCalls))
			for i, tc := range choice.Message.ToolCalls {
				toolCalls[i] = core.ToolCall{
					ID:   tc.ID,
					Name: tc.Function.Name,
					Args: json.RawMessage(tc.Function.Arguments),
					Type: "function",
				}
			}
			aiMsg.ToolCalls = toolCalls
		}

		if resp.Usage != nil {
			aiMsg.UsageMetadata = &core.UsageMetadata{
				InputTokens:  resp.Usage.PromptTokens,
				OutputTokens: resp.Usage.CompletionTokens,
				TotalTokens:  resp.Usage.TotalTokens,
			}
		}

		result.Generations = append(result.Generations, &llms.ChatGeneration{
			Message: aiMsg,
			GenerationInfo: map[string]any{
				"finish_reason": choice.FinishReason,
			},
		})
	}

	return result, nil
}

// streamResponse reads SSE events from the OpenAI streaming response.
func (m *ChatModel) streamResponse(body io.Reader, ch chan<- core.StreamChunk[*core.AIMessage]) {
	scanner := bufio.NewScanner(body)
	var contentBuilder strings.Builder
	var toolCallBuilders = make(map[int]*toolCallBuilder)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			ch <- core.StreamChunk[*core.AIMessage]{Err: fmt.Errorf("failed to parse stream chunk: %w", err)}
			return
		}

		for _, choice := range chunk.Choices {
			delta := choice.Delta

			// Content delta
			if delta.Content != "" {
				contentBuilder.WriteString(delta.Content)
				msg := core.NewAIMessage(delta.Content)
				ch <- core.StreamChunk[*core.AIMessage]{Value: msg}
			}

			// Tool call deltas
			for _, tc := range delta.ToolCalls {
				builder, ok := toolCallBuilders[tc.Index]
				if !ok {
					builder = &toolCallBuilder{}
					toolCallBuilders[tc.Index] = builder
				}
				if tc.ID != "" {
					builder.id = tc.ID
				}
				if tc.Function.Name != "" {
					builder.name = tc.Function.Name
				}
				builder.args += tc.Function.Arguments
			}
		}
	}

	// If we accumulated tool calls, send a final message with them.
	if len(toolCallBuilders) > 0 {
		var toolCalls []core.ToolCall
		for _, builder := range toolCallBuilders {
			toolCalls = append(toolCalls, core.ToolCall{
				ID:   builder.id,
				Name: builder.name,
				Args: json.RawMessage(builder.args),
				Type: "function",
			})
		}
		msg := core.NewAIMessageWithToolCalls(contentBuilder.String(), toolCalls)
		ch <- core.StreamChunk[*core.AIMessage]{Value: msg}
	}
}

type toolCallBuilder struct {
	id   string
	name string
	args string
}

// OpenAI API types

type openAIChatResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []openAIChatChoice `json:"choices"`
	Usage   *openAIUsage       `json:"usage,omitempty"`
}

type openAIChatChoice struct {
	Index        int           `json:"index"`
	Message      openAIChatMsg `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type openAIChatMsg struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	ToolCalls []openAIToolCall `json:"tool_calls,omitempty"`
}

type openAIToolCall struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openAIFunctionCall `json:"function"`
}

type openAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openAIStreamChunk struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []openAIStreamChoice `json:"choices"`
	Usage   *openAIUsage         `json:"usage,omitempty"`
}

type openAIStreamChoice struct {
	Index        int               `json:"index"`
	Delta        openAIStreamDelta `json:"delta"`
	FinishReason *string           `json:"finish_reason,omitempty"`
}

type openAIStreamDelta struct {
	Role      string                 `json:"role,omitempty"`
	Content   string                 `json:"content,omitempty"`
	ToolCalls []openAIStreamToolCall `json:"tool_calls,omitempty"`
}

type openAIStreamToolCall struct {
	Index    int                `json:"index"`
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Function openAIFunctionCall `json:"function"`
}

// Ensure ChatModel implements llms.ChatModel.
var _ llms.ChatModel = (*ChatModel)(nil)
