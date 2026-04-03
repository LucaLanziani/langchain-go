package ollama

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"

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
	structuredSchema map[string]any
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

// GetName returns the model identifier.
func (m *ChatModel) GetName() string {
	return "ChatOllama/" + m.opts.Model
}

// BindTools returns a copy of the model with the given tools bound.
func (m *ChatModel) BindTools(tools ...llms.ToolDefinition) llms.ChatModel {
	cp := *m
	cp.boundTools = append(append([]llms.ToolDefinition(nil), m.boundTools...), tools...)
	return &cp
}

// WithStructuredOutput returns a copy of the model configured for JSON output.
func (m *ChatModel) WithStructuredOutput(schema map[string]any) llms.ChatModel {
	cp := *m
	cp.structuredSchema = schema
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
	respBody, err := doPost(ctx, m.client, m.opts.BaseURL+"/api/chat", req)
	if err != nil {
		return nil, err
	}
	return m.parseResponse(respBody)
}

// Stream sends messages and streams the response token by token.
func (m *ChatModel) Stream(ctx context.Context, input []core.Message, opts ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
	cfg := core.ApplyOptions(opts...)
	req := m.buildRequest(input, cfg, true)

	resp, err := doRawPost(ctx, m.client, m.opts.BaseURL+"/api/chat", req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("ollama: status %d: %s", resp.StatusCode, body)
	}

	ch := make(chan core.StreamChunk[*core.AIMessage], streamBufferSize)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		streamResponse(resp.Body, ch)
	}()

	return core.NewStreamIterator(ch), nil
}

// Batch performs multiple Invoke calls concurrently, preserving input order.
func (m *ChatModel) Batch(ctx context.Context, inputs [][]core.Message, opts ...core.Option) ([]*core.AIMessage, error) {
	type result struct {
		msg *core.AIMessage
		err error
	}
	results := make([]result, len(inputs))
	var wg sync.WaitGroup
	for i, input := range inputs {
		i, input := i, input
		wg.Add(1)
		go func() {
			defer wg.Done()
			msg, err := m.Invoke(ctx, input, opts...)
			results[i] = result{msg, err}
		}()
	}
	wg.Wait()

	msgs := make([]*core.AIMessage, len(inputs))
	for i, r := range results {
		if r.err != nil {
			return nil, fmt.Errorf("ollama: batch item %d: %w", i, r.err)
		}
		msgs[i] = r.msg
	}
	return msgs, nil
}

// buildRequest constructs the Ollama /api/chat request body.
func (m *ChatModel) buildRequest(messages []core.Message, cfg *core.RunnableConfig, stream bool) *chatRequest {
	model := m.opts.Model
	if s, ok := configGet[string](cfg, llms.ConfigKeyModel); ok {
		model = s
	}

	ollamaMsgs := make([]ollamaMessage, 0, len(messages))
	for _, msg := range messages {
		ollamaMsgs = append(ollamaMsgs, messageToOllama(msg))
	}

	format := m.opts.Format
	if m.structuredSchema != nil {
		format = "json"
	}

	mopts := &modelOptions{
		Temperature: m.opts.Temperature,
		TopP:        m.opts.TopP,
		TopK:        m.opts.TopK,
		NumPredict:  m.opts.NumPredict,
		NumCtx:      m.opts.NumCtx,
	}
	if f, ok := configGet[float64](cfg, llms.ConfigKeyTemperature); ok {
		mopts.Temperature = &f
	}
	if n, ok := configGet[int](cfg, llms.ConfigKeyMaxTokens); ok {
		mopts.NumPredict = &n
	}
	if f, ok := configGet[float64](cfg, llms.ConfigKeyTopP); ok {
		mopts.TopP = &f
	}
	mopts.Stop = cfg.Stop
	if len(mopts.Stop) == 0 {
		mopts.Stop = m.opts.Stop
	}

	req := &chatRequest{
		Model:     model,
		Messages:  ollamaMsgs,
		Stream:    stream,
		Format:    format,
		KeepAlive: m.opts.KeepAlive,
	}

	if mopts.Temperature != nil || mopts.TopP != nil || mopts.TopK != nil ||
		mopts.NumPredict != nil || mopts.NumCtx != nil || len(mopts.Stop) > 0 {
		req.Options = mopts
	}

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

	return req
}

// parseResponse parses the non-streaming /api/chat response.
func (m *ChatModel) parseResponse(body []byte) (*llms.ChatResult, error) {
	var resp chatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("ollama: parse response: %w", err)
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
				Message:        aiMsg,
				GenerationInfo: map[string]any{"done_reason": resp.DoneReason},
			},
		},
		LLMOutput: map[string]any{"model": resp.Model},
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

// streamResponse reads NDJSON lines from an Ollama streaming response and sends
// chunks to ch. Content tokens are sent as they arrive. A final chunk with
// accumulated tool calls and/or usage metadata is sent when the stream is done.
func streamResponse(body io.Reader, ch chan<- core.StreamChunk[*core.AIMessage]) {
	scanner := bufio.NewScanner(body)
	var contentBuilder []byte
	var toolCallBuilders []ollamaToolCall

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var chunk streamChunk
		if err := json.Unmarshal(line, &chunk); err != nil {
			ch <- core.StreamChunk[*core.AIMessage]{Err: fmt.Errorf("ollama: parse stream chunk: %w", err)}
			return
		}

		if chunk.Message.Content != "" {
			contentBuilder = append(contentBuilder, chunk.Message.Content...)
			ch <- core.StreamChunk[*core.AIMessage]{Value: core.NewAIMessage(chunk.Message.Content)}
		}

		if len(chunk.Message.ToolCalls) > 0 {
			toolCallBuilders = append(toolCallBuilders, chunk.Message.ToolCalls...)
		}

		if chunk.Done {
			var toolCalls []core.ToolCall
			if len(toolCallBuilders) > 0 {
				toolCalls = make([]core.ToolCall, len(toolCallBuilders))
				for i, tc := range toolCallBuilders {
					toolCalls[i] = core.ToolCall{
						Name: tc.Function.Name,
						Args: tc.Function.Arguments,
						Type: "function",
					}
				}
			}
			total := chunk.PromptEvalCount + chunk.EvalCount
			if len(toolCalls) > 0 || total > 0 {
				msg := core.NewAIMessageWithToolCalls("", toolCalls)
				if total > 0 {
					msg.UsageMetadata = &core.UsageMetadata{
						InputTokens:  chunk.PromptEvalCount,
						OutputTokens: chunk.EvalCount,
						TotalTokens:  total,
					}
				}
				ch <- core.StreamChunk[*core.AIMessage]{Value: msg}
			}
			return
		}
	}

	if err := scanner.Err(); err != nil {
		ch <- core.StreamChunk[*core.AIMessage]{Err: fmt.Errorf("ollama: stream read: %w", err)}
	}
}

// messageToOllama converts a core.Message to the Ollama message format.
func messageToOllama(msg core.Message) ollamaMessage {
	om := ollamaMessage{Content: msg.GetContent()}

	switch msg.GetType() {
	case core.MessageTypeHuman:
		om.Role = "user"
	case core.MessageTypeAI:
		om.Role = "assistant"
		if ai, ok := msg.(*core.AIMessage); ok && len(ai.ToolCalls) > 0 {
			tc := make([]ollamaToolCall, len(ai.ToolCalls))
			for i, c := range ai.ToolCalls {
				tc[i] = ollamaToolCall{Function: ollamaToolCallFunction{Name: c.Name, Arguments: c.Args}}
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

// configGet extracts a typed value from cfg.Configurable.
func configGet[T any](cfg *core.RunnableConfig, key string) (T, bool) {
	v, ok := cfg.Configurable[key]
	if !ok {
		var zero T
		return zero, false
	}
	t, ok := v.(T)
	return t, ok
}

// Ensure ChatModel implements llms.ChatModel.
var _ llms.ChatModel = (*ChatModel)(nil)
