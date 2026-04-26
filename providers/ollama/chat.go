package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
)

// streamBufferSize is the capacity of the internal streaming channel.
const streamBufferSize = 64

// ChatModel is the Ollama chat model implementation.
type ChatModel struct {
	opts             *options
	client           *http.Client
	boundTools       []llms.ToolDefinition
	boundSkills      []llms.SkillDefinition
	structuredSchema map[string]any
	name             string
}

// New creates a new Ollama ChatModel with the provided options.
func New(optFns ...OptionFunc) *ChatModel {
	o := defaultOptions()
	for _, fn := range optFns {
		fn(o)
	}
	return &ChatModel{
		opts:   o,
		client: &http.Client{},
	}
}

// GetName returns the name of this model.
func (m *ChatModel) GetName() string {
	if m.name != "" {
		return m.name
	}
	return "ChatOllama"
}

// BindTools returns a copy of the model with the given tools bound.
func (m *ChatModel) BindTools(tools ...llms.ToolDefinition) llms.ChatModel {
	cp := *m
	cp.boundTools = append(append([]llms.ToolDefinition(nil), m.boundTools...), tools...)
	return &cp
}

// BindSkills returns a copy of the model with the given skills bound.
func (m *ChatModel) BindSkills(skills ...llms.SkillDefinition) llms.ChatModel {
	cp := *m
	cp.boundSkills = append(append([]llms.SkillDefinition(nil), m.boundSkills...), skills...)
	return &cp
}

// WithStructuredOutput returns a copy of the model configured for JSON output.
func (m *ChatModel) WithStructuredOutput(schema map[string]any) llms.ChatModel {
	cp := *m
	cp.structuredSchema = core.CloneMap(schema)
	return &cp
}

// Invoke sends messages to Ollama and returns the AI response.
func (m *ChatModel) Invoke(ctx context.Context, input []core.Message, opts ...core.Option) (*core.AIMessage, error) {
	result, err := m.Generate(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	if len(result.Generations) == 0 {
		return nil, fmt.Errorf("ollama: no generations returned")
	}
	return result.Generations[0].Message, nil
}

// Generate performs a chat completion and returns detailed results.
func (m *ChatModel) Generate(ctx context.Context, messages []core.Message, opts ...core.Option) (*llms.ChatResult, error) {
	cfg := core.ApplyOptions(opts...)
	req := m.buildRequest(messages, cfg, false)

	respBody, err := m.doRequest(ctx, "/api/chat", req)
	if err != nil {
		return nil, err
	}

	return m.parseResponse(respBody)
}

// Stream sends messages and streams the response token by token.
func (m *ChatModel) Stream(ctx context.Context, input []core.Message, opts ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
	cfg := core.ApplyOptions(opts...)
	req := m.buildRequest(input, cfg, true)

	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("ollama: failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.opts.BaseURL+"/api/chat", bytes.NewReader(reqJSON))
	if err != nil {
		return nil, fmt.Errorf("ollama: failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := m.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama: request failed: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("ollama: API error (status %d): %s", resp.StatusCode, string(body))
	}

	ch := make(chan core.StreamChunk[*core.AIMessage], streamBufferSize)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		m.streamResponse(resp.Body, ch)
	}()

	return core.NewStreamIterator(ch), nil
}

// Batch performs multiple Invoke calls sequentially.
func (m *ChatModel) Batch(ctx context.Context, inputs [][]core.Message, opts ...core.Option) ([]*core.AIMessage, error) {
	return core.Batch(ctx, inputs, opts, func(ctx context.Context, input []core.Message, opts ...core.Option) (*core.AIMessage, error) {
		result, err := m.Invoke(ctx, input, opts...)
		if err != nil {
			return nil, fmt.Errorf("ollama: %w", err)
		}
		return result, nil
	})
}

// buildRequest constructs the Ollama /api/chat request body.
func (m *ChatModel) buildRequest(messages []core.Message, cfg *core.RunnableConfig, stream bool) *chatRequest {
	model := m.opts.Model
	if v, ok := cfg.Configurable[llms.ConfigKeyModel]; ok {
		if s, ok := v.(string); ok {
			model = s
		}
	}

	ollamaMsgs := make([]ollamaMessage, 0, len(messages))
	for _, msg := range messages {
		ollamaMsgs = append(ollamaMsgs, messageToOllama(msg))
	}

	req := &chatRequest{
		Model:     model,
		Messages:  ollamaMsgs,
		Stream:    stream,
		KeepAlive: m.opts.KeepAlive,
	}

	// Format
	format := m.opts.Format
	if m.structuredSchema != nil {
		format = "json"
	}
	req.Format = format

	// Build model options
	mopts := &modelOptions{
		Temperature: m.opts.Temperature,
		TopP:        m.opts.TopP,
		TopK:        m.opts.TopK,
		NumPredict:  m.opts.NumPredict,
		NumCtx:      m.opts.NumCtx,
	}

	// Override with per-call options
	if temp, ok := cfg.Configurable[llms.ConfigKeyTemperature]; ok {
		if f, ok := temp.(float64); ok {
			mopts.Temperature = &f
		}
	}
	if mt, ok := cfg.Configurable[llms.ConfigKeyMaxTokens]; ok {
		if n, ok := mt.(int); ok {
			mopts.NumPredict = &n
		}
	}
	if tp, ok := cfg.Configurable[llms.ConfigKeyTopP]; ok {
		if f, ok := tp.(float64); ok {
			mopts.TopP = &f
		}
	}

	stop := cfg.Stop
	if len(stop) == 0 {
		stop = m.opts.Stop
	}
	mopts.Stop = stop

	// Only include options block if at least one field is set
	if mopts.Temperature != nil || mopts.TopP != nil || mopts.TopK != nil ||
		mopts.NumPredict != nil || mopts.NumCtx != nil || len(mopts.Stop) > 0 {
		req.Options = mopts
	}

	// Tools
	if len(m.boundTools) > 0 {
		tools := make([]ollamaTool, len(m.boundTools))
		for i, t := range m.boundTools {
			tools[i] = ollamaTool{
				Type: "function",
				Function: ollamaToolFunction{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.Parameters,
				},
			}
		}
		req.Tools = tools
	}

	applySkills(req, m.boundSkills)

	return req
}

func applySkills(req *chatRequest, skills []llms.SkillDefinition) {
	_ = req
	_ = skills
}

// doRequest sends an HTTP POST request and returns the response body.
func (m *ChatModel) doRequest(ctx context.Context, path string, body any) ([]byte, error) {
	reqJSON, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("ollama: failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.opts.BaseURL+path, bytes.NewReader(reqJSON))
	if err != nil {
		return nil, fmt.Errorf("ollama: failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama: request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("ollama: failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama: API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

// parseResponse parses the non-streaming /api/chat response.
func (m *ChatModel) parseResponse(body []byte) (*llms.ChatResult, error) {
	var resp chatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("ollama: failed to parse response: %w", err)
	}

	aiMsg := core.NewAIMessage(resp.Message.Content)
	aiMsg.ResponseMetadata = map[string]any{
		"model":       resp.Model,
		"done_reason": resp.DoneReason,
	}

	if len(resp.Message.ToolCalls) > 0 {
		toolCalls := make([]core.ToolCall, len(resp.Message.ToolCalls))
		for i, tc := range resp.Message.ToolCalls {
			toolCalls[i] = core.ToolCall{
				Name: tc.Function.Name,
				Args: tc.Function.Arguments,
				Type: "function",
			}
		}
		aiMsg.ToolCalls = toolCalls
	}

	totalTokens := resp.PromptEvalCount + resp.EvalCount
	if totalTokens > 0 {
		aiMsg.UsageMetadata = &core.UsageMetadata{
			InputTokens:  resp.PromptEvalCount,
			OutputTokens: resp.EvalCount,
			TotalTokens:  totalTokens,
		}
	}

	result := &llms.ChatResult{
		Generations: []*llms.ChatGeneration{
			{
				Message: aiMsg,
				GenerationInfo: map[string]any{
					"done_reason": resp.DoneReason,
				},
			},
		},
		LLMOutput: map[string]any{
			"model": resp.Model,
		},
	}

	if totalTokens > 0 {
		result.LLMOutput["token_usage"] = llms.TokenUsage{
			PromptTokens:     resp.PromptEvalCount,
			CompletionTokens: resp.EvalCount,
			TotalTokens:      totalTokens,
		}
	}

	return result, nil
}

// streamResponse reads NDJSON lines from an Ollama streaming response.
func (m *ChatModel) streamResponse(body io.Reader, ch chan<- core.StreamChunk[*core.AIMessage]) {
	scanner := bufio.NewScanner(body)
	var contentBuilder strings.Builder
	var toolCallBuilders []ollamaToolCall

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var chunk streamChunk
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			ch <- core.StreamChunk[*core.AIMessage]{Err: fmt.Errorf("ollama: failed to parse stream chunk: %w", err)}
			return
		}

		if chunk.Message.Content != "" {
			contentBuilder.WriteString(chunk.Message.Content)
			msg := core.NewAIMessage(chunk.Message.Content)
			ch <- core.StreamChunk[*core.AIMessage]{Value: msg}
		}

		if len(chunk.Message.ToolCalls) > 0 {
			toolCallBuilders = append(toolCallBuilders, chunk.Message.ToolCalls...)
		}

		if chunk.Done {
			// Send a final message with accumulated tool calls, if any.
			if len(toolCallBuilders) > 0 {
				toolCalls := make([]core.ToolCall, len(toolCallBuilders))
				for i, tc := range toolCallBuilders {
					toolCalls[i] = core.ToolCall{
						Name: tc.Function.Name,
						Args: tc.Function.Arguments,
						Type: "function",
					}
				}
				msg := core.NewAIMessageWithToolCalls(contentBuilder.String(), toolCalls)
				ch <- core.StreamChunk[*core.AIMessage]{Value: msg}
			}
			return
		}
	}

	if err := scanner.Err(); err != nil {
		ch <- core.StreamChunk[*core.AIMessage]{Err: fmt.Errorf("ollama: stream read error: %w", err)}
	}
}

// messageToOllama converts a core.Message to the Ollama message format.
func messageToOllama(msg core.Message) ollamaMessage {
	om := ollamaMessage{
		Content: msg.GetContent(),
	}

	switch msg.GetType() {
	case core.MessageTypeHuman:
		om.Role = "user"
	case core.MessageTypeAI:
		om.Role = "assistant"
		if ai, ok := msg.(*core.AIMessage); ok && len(ai.ToolCalls) > 0 {
			tc := make([]ollamaToolCall, len(ai.ToolCalls))
			for i, c := range ai.ToolCalls {
				tc[i] = ollamaToolCall{
					Function: ollamaToolCallFunction{
						Name:      c.Name,
						Arguments: c.Args,
					},
				}
			}
			om.ToolCalls = tc
		}
	case core.MessageTypeSystem:
		om.Role = "system"
	case core.MessageTypeTool:
		om.Role = "tool"
		if tm, ok := msg.(*core.ToolMessage); ok {
			om.ToolCallID = tm.ToolCallID
		}
	default:
		om.Role = "user"
	}

	return om
}

// Ensure ChatModel implements llms.ChatModel.
var _ llms.ChatModel = (*ChatModel)(nil)
