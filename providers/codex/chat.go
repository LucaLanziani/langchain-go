package codex

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
)

// ChatModel talks to ChatGPT's Codex backend using the user's CLI subscription.
type ChatModel struct {
	opts             *Options
	auth             *AuthManager
	client           *http.Client
	streamClient     *http.Client
	boundTools       []llms.ToolDefinition
	boundSkills      []llms.SkillDefinition
	structuredSchema map[string]any
	sessionID        string
	name             string
}

var _ llms.ChatModel = (*ChatModel)(nil)

// New builds a ChatModel. It loads the auth file eagerly so misconfiguration
// surfaces during construction, but defers token refresh until the first request.
func New(optFns ...OptionFunc) (*ChatModel, error) {
	opts := DefaultOptions()
	for _, fn := range optFns {
		fn(opts)
	}

	auth, err := NewAuthManager(opts.AuthFile)
	if err != nil {
		return nil, err
	}

	timeout := time.Duration(opts.HTTPTimeout) * time.Second

	return &ChatModel{
		opts:         opts,
		auth:         auth,
		client:       &http.Client{Timeout: timeout},
		streamClient: &http.Client{}, // no timeout for SSE
		sessionID:    newUUID(),
	}, nil
}

// GetName returns the chat model name.
func (m *ChatModel) GetName() string {
	if m.name != "" {
		return m.name
	}
	return "ChatCodex"
}

// BindTools returns a copy of the model with tool definitions attached.
func (m *ChatModel) BindTools(tools ...llms.ToolDefinition) llms.ChatModel {
	cp := *m
	cp.boundTools = append(append([]llms.ToolDefinition(nil), m.boundTools...), tools...)
	return &cp
}

// BindSkills returns a copy of the model with skills bound. Codex's Responses
// API has no native skill concept, so bound skills are forwarded as appended
// instructions instead.
func (m *ChatModel) BindSkills(skills ...llms.SkillDefinition) llms.ChatModel {
	cp := *m
	cp.boundSkills = append(append([]llms.SkillDefinition(nil), m.boundSkills...), skills...)
	return &cp
}

// WithStructuredOutput requests JSON output that conforms to the given schema.
// The Codex backend does not advertise json_schema response_format on the
// ChatGPT path, so the schema is injected into the instructions.
func (m *ChatModel) WithStructuredOutput(schema map[string]any) llms.ChatModel {
	cp := *m
	cp.structuredSchema = core.CloneMap(schema)
	return &cp
}

// Invoke sends messages and returns the assistant's reply.
func (m *ChatModel) Invoke(ctx context.Context, input []core.Message, opts ...core.Option) (*core.AIMessage, error) {
	result, err := m.Generate(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	if len(result.Generations) == 0 {
		return nil, errors.New("codex: no generations returned")
	}
	return result.Generations[0].Message, nil
}

// Generate performs a chat completion and returns the full result. It uses the
// streaming endpoint under the hood because Codex's Responses API is
// streaming-first.
func (m *ChatModel) Generate(ctx context.Context, messages []core.Message, opts ...core.Option) (*llms.ChatResult, error) {
	iter, err := m.Stream(ctx, messages, opts...)
	if err != nil {
		return nil, err
	}
	defer iter.Close()

	var (
		content      strings.Builder
		toolCalls    []core.ToolCall
		usage        *core.UsageMetadata
		finishReason string
	)

	for {
		chunk, ok, err := iter.Next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		content.WriteString(chunk.GetContent())
		if len(chunk.ToolCalls) > 0 {
			toolCalls = append(toolCalls, chunk.ToolCalls...)
		}
		if chunk.UsageMetadata != nil {
			usage = chunk.UsageMetadata
		}
		if chunk.ResponseMetadata != nil {
			if fr, ok := chunk.ResponseMetadata["finish_reason"].(string); ok && fr != "" {
				finishReason = fr
			}
		}
	}

	var aiMsg *core.AIMessage
	if len(toolCalls) > 0 {
		aiMsg = core.NewAIMessageWithToolCalls(content.String(), toolCalls)
	} else {
		aiMsg = core.NewAIMessage(content.String())
	}
	aiMsg.UsageMetadata = usage
	if finishReason != "" {
		aiMsg.ResponseMetadata = map[string]any{"finish_reason": finishReason}
	}

	result := &llms.ChatResult{
		LLMOutput: map[string]any{
			"provider": "codex",
			"model":    m.opts.Model,
		},
		Generations: []*llms.ChatGeneration{
			{
				Message: aiMsg,
				GenerationInfo: map[string]any{
					"provider":      "codex",
					"finish_reason": finishReason,
				},
			},
		},
	}
	if usage != nil {
		result.LLMOutput["token_usage"] = llms.TokenUsage{
			PromptTokens:     usage.InputTokens,
			CompletionTokens: usage.OutputTokens,
			TotalTokens:      usage.TotalTokens,
		}
	}
	return result, nil
}

// Stream opens an SSE stream against the Codex Responses API and decodes events
// into AIMessage chunks.
func (m *ChatModel) Stream(ctx context.Context, input []core.Message, opts ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
	cfg := core.ApplyOptions(opts...)
	body, err := m.buildRequestBody(input, cfg)
	if err != nil {
		return nil, err
	}
	reqJSON, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("codex: marshal request: %w", err)
	}

	accessToken, accountID, err := m.auth.AccessToken(ctx)
	if err != nil {
		return nil, err
	}

	url := strings.TrimRight(m.opts.BaseURL, "/") + "/responses"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqJSON))
	if err != nil {
		return nil, fmt.Errorf("codex: build request: %w", err)
	}
	m.setHeaders(req, accessToken, accountID)

	resp, err := m.streamClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("codex: request failed: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("codex: Responses API error (status %d): %s", resp.StatusCode, string(errBody))
	}

	ch := make(chan core.StreamChunk[*core.AIMessage], 64)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		decodeSSE(ctx, resp.Body, ch)
	}()
	return core.NewStreamIterator(ch), nil
}

// Batch performs multiple chat completions sequentially.
func (m *ChatModel) Batch(ctx context.Context, inputs [][]core.Message, opts ...core.Option) ([]*core.AIMessage, error) {
	return core.Batch(ctx, inputs, opts, m.Invoke)
}

// buildRequestBody assembles a Responses API request body from langchain messages.
func (m *ChatModel) buildRequestBody(messages []core.Message, cfg *core.RunnableConfig) (map[string]any, error) {
	model := m.opts.Model
	if v, ok := cfg.Configurable[llms.ConfigKeyModel]; ok {
		if s, ok := v.(string); ok && s != "" {
			model = s
		}
	}

	instructions, items := splitInstructions(messages)
	if m.opts.Instructions != "" {
		if instructions != "" {
			instructions = m.opts.Instructions + "\n\n" + instructions
		} else {
			instructions = m.opts.Instructions
		}
	}
	if m.structuredSchema != nil {
		schemaJSON, err := json.Marshal(m.structuredSchema)
		if err != nil {
			return nil, fmt.Errorf("codex: marshal structured schema: %w", err)
		}
		instr := fmt.Sprintf("You must respond with valid JSON that conforms to this JSON schema:\n%s", string(schemaJSON))
		if instructions != "" {
			instructions = instructions + "\n\n" + instr
		} else {
			instructions = instr
		}
	}
	for _, skill := range m.boundSkills {
		if skill.Instructions == "" {
			continue
		}
		section := fmt.Sprintf("Skill: %s\n%s", skill.Name, skill.Instructions)
		if instructions != "" {
			instructions = instructions + "\n\n" + section
		} else {
			instructions = section
		}
	}

	body := map[string]any{
		"model":               model,
		"input":               items,
		"stream":              true,
		"store":               false,
		"parallel_tool_calls": false,
		"include":             []string{"reasoning.encrypted_content"},
	}
	if instructions != "" {
		body["instructions"] = instructions
	}
	if tools := m.toolDefinitions(); len(tools) > 0 {
		body["tools"] = tools
		body["tool_choice"] = "auto"
	}

	if m.opts.ReasoningEffort != "" || m.opts.ReasoningSummary != "" {
		reasoning := map[string]any{}
		if m.opts.ReasoningEffort != "" {
			reasoning["effort"] = m.opts.ReasoningEffort
		}
		if m.opts.ReasoningSummary != "" {
			reasoning["summary"] = m.opts.ReasoningSummary
		}
		body["reasoning"] = reasoning
	}

	if m.opts.MaxOutputTokens != nil {
		body["max_output_tokens"] = *m.opts.MaxOutputTokens
	}
	if m.opts.PromptCacheKey != "" {
		body["prompt_cache_key"] = m.opts.PromptCacheKey
	} else {
		body["prompt_cache_key"] = m.sessionID
	}
	return body, nil
}

func (m *ChatModel) toolDefinitions() []map[string]any {
	if len(m.boundTools) == 0 {
		return nil
	}
	out := make([]map[string]any, 0, len(m.boundTools))
	for _, t := range m.boundTools {
		out = append(out, map[string]any{
			"type":        "function",
			"name":        t.Name,
			"description": t.Description,
			"parameters":  t.Parameters,
		})
	}
	return out
}

// setHeaders configures the headers the Codex backend expects.
func (m *ChatModel) setHeaders(req *http.Request, accessToken, accountID string) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Authorization", "Bearer "+accessToken)
	if accountID != "" {
		req.Header.Set("ChatGPT-Account-ID", accountID)
	}
	req.Header.Set("OpenAI-Beta", "responses=experimental")
	req.Header.Set("originator", m.opts.Originator)
	if m.opts.UserAgent != "" {
		req.Header.Set("User-Agent", m.opts.UserAgent)
	}
	req.Header.Set("session_id", m.sessionID)
}

// splitInstructions pulls system messages out into a single instructions string
// and converts the remaining messages into Responses-API input items.
func splitInstructions(messages []core.Message) (string, []map[string]any) {
	var systemParts []string
	items := make([]map[string]any, 0, len(messages))
	for _, msg := range messages {
		switch msg.GetType() {
		case core.MessageTypeSystem:
			if c := msg.GetContent(); c != "" {
				systemParts = append(systemParts, c)
			}
		case core.MessageTypeHuman:
			items = append(items, map[string]any{
				"type": "message",
				"role": "user",
				"content": []map[string]any{
					{"type": "input_text", "text": msg.GetContent()},
				},
			})
		case core.MessageTypeAI:
			content := msg.GetContent()
			if content != "" {
				items = append(items, map[string]any{
					"type": "message",
					"role": "assistant",
					"content": []map[string]any{
						{"type": "output_text", "text": content},
					},
				})
			}
			if ai, ok := msg.(*core.AIMessage); ok {
				for _, tc := range ai.ToolCalls {
					items = append(items, map[string]any{
						"type":      "function_call",
						"call_id":   tc.ID,
						"name":      tc.Name,
						"arguments": string(tc.Args),
					})
				}
			}
		case core.MessageTypeTool:
			if tm, ok := msg.(*core.ToolMessage); ok {
				items = append(items, map[string]any{
					"type":    "function_call_output",
					"call_id": tm.ToolCallID,
					"output":  msg.GetContent(),
				})
			} else {
				items = append(items, map[string]any{
					"type":   "function_call_output",
					"output": msg.GetContent(),
				})
			}
		case core.MessageTypeFunction:
			items = append(items, map[string]any{
				"type":    "function_call_output",
				"call_id": msg.GetName(),
				"output":  msg.GetContent(),
			})
		default:
			items = append(items, map[string]any{
				"type": "message",
				"role": "user",
				"content": []map[string]any{
					{"type": "input_text", "text": msg.GetContent()},
				},
			})
		}
	}
	return strings.Join(systemParts, "\n\n"), items
}

// decodeSSE parses the Codex Responses API event stream.
func decodeSSE(ctx context.Context, body io.Reader, ch chan<- core.StreamChunk[*core.AIMessage]) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	var eventType string
	var dataBuf strings.Builder

	dispatch := func() {
		defer func() {
			eventType = ""
			dataBuf.Reset()
		}()
		if dataBuf.Len() == 0 {
			return
		}
		raw := dataBuf.String()
		if raw == "[DONE]" {
			return
		}
		var evt sseEnvelope
		if err := json.Unmarshal([]byte(raw), &evt); err != nil {
			ch <- core.StreamChunk[*core.AIMessage]{Err: fmt.Errorf("codex: decode SSE event: %w", err)}
			return
		}
		if evt.Type == "" {
			evt.Type = eventType
		}
		handleEvent(evt, ch)
	}

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			ch <- core.StreamChunk[*core.AIMessage]{Err: ctx.Err()}
			return
		default:
		}

		line := scanner.Text()
		if line == "" {
			dispatch()
			continue
		}
		switch {
		case strings.HasPrefix(line, "event:"):
			eventType = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			if dataBuf.Len() > 0 {
				dataBuf.WriteByte('\n')
			}
			dataBuf.WriteString(strings.TrimPrefix(line, "data:"))
			// Trim a single leading space if present (SSE spec).
			if dataBuf.Len() > 0 {
				s := dataBuf.String()
				if strings.HasPrefix(s, " ") {
					dataBuf.Reset()
					dataBuf.WriteString(strings.TrimPrefix(s, " "))
				}
			}
		case strings.HasPrefix(line, ":"):
			// Comment / keepalive — ignore.
		}
	}
	dispatch()
	if err := scanner.Err(); err != nil {
		ch <- core.StreamChunk[*core.AIMessage]{Err: fmt.Errorf("codex: read SSE: %w", err)}
	}
}

func handleEvent(evt sseEnvelope, ch chan<- core.StreamChunk[*core.AIMessage]) {
	switch evt.Type {
	case "response.output_text.delta":
		if evt.Delta != "" {
			ch <- core.StreamChunk[*core.AIMessage]{Value: core.NewAIMessage(evt.Delta)}
		}
	case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
		if evt.Delta != "" {
			msg := core.NewAIMessage("")
			msg.AdditionalKwargs = map[string]any{"thinking": evt.Delta}
			ch <- core.StreamChunk[*core.AIMessage]{Value: msg}
		}
	case "response.output_item.done":
		if evt.Item == nil {
			return
		}
		if evt.Item.Type == "function_call" {
			toolCall := core.ToolCall{
				ID:   evt.Item.CallID,
				Name: evt.Item.Name,
				Args: json.RawMessage(evt.Item.Arguments),
				Type: "function",
			}
			ch <- core.StreamChunk[*core.AIMessage]{
				Value: core.NewAIMessageWithToolCalls("", []core.ToolCall{toolCall}),
			}
		}
	case "response.completed":
		if evt.Response == nil {
			return
		}
		msg := core.NewAIMessage("")
		if evt.Response.Usage != nil {
			msg.UsageMetadata = &core.UsageMetadata{
				InputTokens:  evt.Response.Usage.InputTokens,
				OutputTokens: evt.Response.Usage.OutputTokens,
				TotalTokens:  evt.Response.Usage.TotalTokens,
			}
		}
		msg.ResponseMetadata = map[string]any{"finish_reason": "stop"}
		ch <- core.StreamChunk[*core.AIMessage]{Value: msg}
	case "response.failed":
		errMsg := "codex: response.failed"
		if evt.Response != nil && evt.Response.Error != nil {
			errMsg = fmt.Sprintf("codex: %s: %s", evt.Response.Error.Code, evt.Response.Error.Message)
		}
		ch <- core.StreamChunk[*core.AIMessage]{Err: errors.New(errMsg)}
	case "response.incomplete":
		msg := core.NewAIMessage("")
		msg.ResponseMetadata = map[string]any{"finish_reason": "incomplete"}
		ch <- core.StreamChunk[*core.AIMessage]{Value: msg}
	}
}

type sseEnvelope struct {
	Type     string           `json:"type"`
	Delta    string           `json:"delta,omitempty"`
	Item     *sseOutputItem   `json:"item,omitempty"`
	Response *sseResponseBody `json:"response,omitempty"`
}

type sseOutputItem struct {
	Type      string `json:"type"`
	ID        string `json:"id,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type sseResponseBody struct {
	ID    string           `json:"id,omitempty"`
	Usage *sseUsage        `json:"usage,omitempty"`
	Error *sseResponseErr  `json:"error,omitempty"`
}

type sseUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type sseResponseErr struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// newUUID generates a RFC 4122 v4 UUID using crypto/rand. We avoid pulling in
// an external dependency for a single value.
func newUUID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
