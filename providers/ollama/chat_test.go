package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
)

// ---------- helpers ----------

func newTestModel(srv *httptest.Server) *ChatModel {
	return New(WithBaseURL(srv.URL), WithModel("test-model"))
}

func jsonResponse(t *testing.T, w http.ResponseWriter, v any) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(v); err != nil {
		t.Fatalf("failed to encode response: %v", err)
	}
}

// ---------- options ----------

func TestDefaultOptions(t *testing.T) {
	o := defaultOptions()
	if o.BaseURL != "http://localhost:11434" {
		t.Errorf("expected default BaseURL, got %q", o.BaseURL)
	}
	if o.Model != "llama3.1" {
		t.Errorf("expected default Model 'llama3.1', got %q", o.Model)
	}
}

func TestOptions(t *testing.T) {
	temp := 0.7
	topP := 0.9
	topK := 40
	numPredict := 100
	numCtx := 2048

	o := defaultOptions()
	WithModel("mistral")(o)
	WithBaseURL("http://myhost:11434")(o)
	WithTemperature(temp)(o)
	WithTopP(topP)(o)
	WithTopK(topK)(o)
	WithNumPredict(numPredict)(o)
	WithNumCtx(numCtx)(o)
	WithStop([]string{"</s>"})(o)
	WithFormat("json")(o)
	WithKeepAlive("5m")(o)

	if o.Model != "mistral" {
		t.Errorf("expected model 'mistral', got %q", o.Model)
	}
	if o.BaseURL != "http://myhost:11434" {
		t.Errorf("unexpected BaseURL: %q", o.BaseURL)
	}
	if o.Temperature == nil || *o.Temperature != temp {
		t.Errorf("unexpected Temperature: %v", o.Temperature)
	}
	if o.TopP == nil || *o.TopP != topP {
		t.Errorf("unexpected TopP: %v", o.TopP)
	}
	if o.TopK == nil || *o.TopK != topK {
		t.Errorf("unexpected TopK: %v", o.TopK)
	}
	if o.NumPredict == nil || *o.NumPredict != numPredict {
		t.Errorf("unexpected NumPredict: %v", o.NumPredict)
	}
	if o.NumCtx == nil || *o.NumCtx != numCtx {
		t.Errorf("unexpected NumCtx: %v", o.NumCtx)
	}
	if len(o.Stop) != 1 || o.Stop[0] != "</s>" {
		t.Errorf("unexpected Stop: %v", o.Stop)
	}
	if o.Format != "json" {
		t.Errorf("unexpected Format: %q", o.Format)
	}
	if o.KeepAlive != "5m" {
		t.Errorf("unexpected KeepAlive: %q", o.KeepAlive)
	}
}

// ---------- message conversion ----------

func TestMessageToOllama(t *testing.T) {
	tests := []struct {
		name     string
		msg      core.Message
		wantRole string
	}{
		{"human", core.NewHumanMessage("hi"), "user"},
		{"ai", core.NewAIMessage("hello"), "assistant"},
		{"system", core.NewSystemMessage("be helpful"), "system"},
		{"tool", core.NewToolMessage("result", "call-1"), "tool"},
		{"unknown", core.NewGenericMessage("custom", "text"), "user"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			om := messageToOllama(tt.msg)
			if om.Role != tt.wantRole {
				t.Errorf("expected role %q, got %q", tt.wantRole, om.Role)
			}
			if om.Content != tt.msg.GetContent() {
				t.Errorf("expected content %q, got %q", tt.msg.GetContent(), om.Content)
			}
		})
	}
}

func TestMessageToOllama_ToolCallID(t *testing.T) {
	tm := core.NewToolMessage("result", "call-xyz")
	om := messageToOllama(tm)
	if om.ToolCallID != "call-xyz" {
		t.Errorf("expected tool_call_id 'call-xyz', got %q", om.ToolCallID)
	}
}

func TestMessageToOllama_AIWithToolCalls(t *testing.T) {
	ai := core.NewAIMessageWithToolCalls("", []core.ToolCall{
		{Name: "get_weather", Args: json.RawMessage(`{"city":"Rome"}`), Type: "function"},
	})
	om := messageToOllama(ai)
	if om.Role != "assistant" {
		t.Errorf("expected role 'assistant', got %q", om.Role)
	}
	if len(om.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(om.ToolCalls))
	}
	if om.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("unexpected tool call name: %q", om.ToolCalls[0].Function.Name)
	}
}

// ---------- ChatModel.GetName ----------

func TestGetName(t *testing.T) {
	m := New()
	if m.GetName() != "ChatOllama" {
		t.Errorf("expected 'ChatOllama', got %q", m.GetName())
	}
	m.name = "MyModel"
	if m.GetName() != "MyModel" {
		t.Errorf("expected 'MyModel', got %q", m.GetName())
	}
}

// ---------- ChatModel.Invoke ----------

func TestInvoke(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		jsonResponse(t, w, chatResponse{
			Model:           "test-model",
			Done:            true,
			DoneReason:      "stop",
			Message:         ollamaMessage{Role: "assistant", Content: "Hello, World!"},
			PromptEvalCount: 10,
			EvalCount:       5,
		})
	}))
	defer srv.Close()

	m := newTestModel(srv)
	msg, err := m.Invoke(context.Background(), []core.Message{
		core.NewHumanMessage("hi"),
	})
	if err != nil {
		t.Fatalf("Invoke error: %v", err)
	}
	if msg.GetContent() != "Hello, World!" {
		t.Errorf("unexpected content: %q", msg.GetContent())
	}
	if msg.UsageMetadata == nil {
		t.Fatal("expected usage metadata")
	}
	if msg.UsageMetadata.InputTokens != 10 {
		t.Errorf("expected 10 input tokens, got %d", msg.UsageMetadata.InputTokens)
	}
	if msg.UsageMetadata.OutputTokens != 5 {
		t.Errorf("expected 5 output tokens, got %d", msg.UsageMetadata.OutputTokens)
	}
}

func TestInvoke_APIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"model not found"}`, http.StatusNotFound)
	}))
	defer srv.Close()

	m := newTestModel(srv)
	_, err := m.Invoke(context.Background(), []core.Message{core.NewHumanMessage("hi")})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "404") {
		t.Errorf("expected 404 in error, got: %v", err)
	}
}

// ---------- ChatModel.Generate ----------

func TestGenerate_SendsCorrectRequest(t *testing.T) {
	var captured chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		jsonResponse(t, w, chatResponse{
			Model:   "test-model",
			Done:    true,
			Message: ollamaMessage{Role: "assistant", Content: "ok"},
		})
	}))
	defer srv.Close()

	m := newTestModel(srv)
	_, err := m.Generate(context.Background(), []core.Message{
		core.NewSystemMessage("system"),
		core.NewHumanMessage("user"),
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}
	if captured.Model != "test-model" {
		t.Errorf("expected model 'test-model', got %q", captured.Model)
	}
	if len(captured.Messages) != 2 {
		t.Errorf("expected 2 messages, got %d", len(captured.Messages))
	}
	if captured.Stream {
		t.Error("stream should be false for Generate")
	}
}

func TestGenerate_WithTemperatureOption(t *testing.T) {
	var captured chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		jsonResponse(t, w, chatResponse{Done: true, Message: ollamaMessage{Role: "assistant", Content: "ok"}})
	}))
	defer srv.Close()

	m := newTestModel(srv)
	_, err := m.Generate(context.Background(), []core.Message{core.NewHumanMessage("hi")},
		llms.WithTemperature(0.5),
	)
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}
	if captured.Options == nil {
		t.Fatal("expected options block")
	}
	if captured.Options.Temperature == nil || *captured.Options.Temperature != 0.5 {
		t.Errorf("expected temperature 0.5, got %v", captured.Options.Temperature)
	}
}

// ---------- ChatModel.Stream ----------

func TestStream(t *testing.T) {
	chunks := []streamChunk{
		{Model: "test-model", Message: ollamaMessage{Role: "assistant", Content: "Hello"}, Done: false},
		{Model: "test-model", Message: ollamaMessage{Role: "assistant", Content: ", "}, Done: false},
		{Model: "test-model", Message: ollamaMessage{Role: "assistant", Content: "World!"}, Done: false},
		{Model: "test-model", Message: ollamaMessage{Role: "assistant", Content: ""}, Done: true, PromptEvalCount: 5, EvalCount: 3},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "%s\n", data)
		}
	}))
	defer srv.Close()

	m := newTestModel(srv)
	iter, err := m.Stream(context.Background(), []core.Message{core.NewHumanMessage("hi")})
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	var parts []string
	for {
		msg, ok, err := iter.Next()
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		if !ok {
			break
		}
		parts = append(parts, msg.GetContent())
	}

	if len(parts) == 0 {
		t.Fatal("expected at least one chunk")
	}
	full := strings.Join(parts, "")
	if full != "Hello, World!" {
		t.Errorf("expected 'Hello, World!', got %q", full)
	}
}

func TestStream_ToolCalls(t *testing.T) {
	toolArgs := json.RawMessage(`{"city":"Paris"}`)
	chunks := []streamChunk{
		{
			Model: "test-model",
			Message: ollamaMessage{
				Role:    "assistant",
				Content: "",
				ToolCalls: []ollamaToolCall{
					{Function: ollamaToolCallFunction{Name: "get_weather", Arguments: toolArgs}},
				},
			},
			Done: false,
		},
		{Model: "test-model", Message: ollamaMessage{Role: "assistant", Content: ""}, Done: true},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "%s\n", data)
		}
	}))
	defer srv.Close()

	m := newTestModel(srv)
	iter, err := m.Stream(context.Background(), []core.Message{core.NewHumanMessage("weather in Paris?")})
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	var finalMsg *core.AIMessage
	for {
		msg, ok, err := iter.Next()
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		if !ok {
			break
		}
		if len(msg.ToolCalls) > 0 {
			finalMsg = msg
		}
	}

	if finalMsg == nil {
		t.Fatal("expected a message with tool calls")
	}
	if len(finalMsg.ToolCalls) != 1 || finalMsg.ToolCalls[0].Name != "get_weather" {
		t.Errorf("unexpected tool calls: %+v", finalMsg.ToolCalls)
	}
}

// ---------- ChatModel.Batch ----------

func TestBatch(t *testing.T) {
	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		jsonResponse(t, w, chatResponse{
			Done:    true,
			Message: ollamaMessage{Role: "assistant", Content: fmt.Sprintf("response %d", callCount)},
		})
	}))
	defer srv.Close()

	m := newTestModel(srv)
	results, err := m.Batch(context.Background(), [][]core.Message{
		{core.NewHumanMessage("msg1")},
		{core.NewHumanMessage("msg2")},
		{core.NewHumanMessage("msg3")},
	})
	if err != nil {
		t.Fatalf("Batch error: %v", err)
	}
	if len(results) != 3 {
		t.Errorf("expected 3 results, got %d", len(results))
	}
	if callCount != 3 {
		t.Errorf("expected 3 API calls, got %d", callCount)
	}
}

// ---------- ChatModel.BindTools ----------

func TestBindTools(t *testing.T) {
	var captured chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		jsonResponse(t, w, chatResponse{
			Done:    true,
			Message: ollamaMessage{Role: "assistant", Content: ""},
		})
	}))
	defer srv.Close()

	tool := llms.ToolDefinition{
		Name:        "get_weather",
		Description: "Get current weather",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
		},
	}

	m := newTestModel(srv)
	bound := m.BindTools(tool)
	_, err := bound.Invoke(context.Background(), []core.Message{core.NewHumanMessage("weather?")})
	if err != nil {
		t.Fatalf("Invoke error: %v", err)
	}
	if len(captured.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(captured.Tools))
	}
	if captured.Tools[0].Function.Name != "get_weather" {
		t.Errorf("unexpected tool name: %q", captured.Tools[0].Function.Name)
	}
}

func TestBindToolsDoesNotAliasDerivedModels(t *testing.T) {
	m := &ChatModel{opts: defaultOptions(), boundTools: make([]llms.ToolDefinition, 1, 4)}
	m.boundTools[0] = llms.ToolDefinition{Name: "base"}

	left := m.BindTools(llms.ToolDefinition{Name: "left"}).(*ChatModel)
	right := m.BindTools(llms.ToolDefinition{Name: "right"}).(*ChatModel)

	if left.boundTools[1].Name != "left" {
		t.Fatalf("expected left tool to remain isolated, got %q", left.boundTools[1].Name)
	}
	if right.boundTools[1].Name != "right" {
		t.Fatalf("expected right tool to remain isolated, got %q", right.boundTools[1].Name)
	}
}

// ---------- ChatModel.WithStructuredOutput ----------

func TestWithStructuredOutput(t *testing.T) {
	var captured chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		jsonResponse(t, w, chatResponse{
			Done:    true,
			Message: ollamaMessage{Role: "assistant", Content: `{"answer":"42"}`},
		})
	}))
	defer srv.Close()

	m := newTestModel(srv)
	structured := m.WithStructuredOutput(map[string]any{"type": "object"})
	_, err := structured.Invoke(context.Background(), []core.Message{core.NewHumanMessage("answer?")})
	if err != nil {
		t.Fatalf("Invoke error: %v", err)
	}
	if captured.Format != "json" {
		t.Errorf("expected format 'json', got %q", captured.Format)
	}
}

func TestWithStructuredOutputClonesSchema(t *testing.T) {
	m := &ChatModel{opts: defaultOptions()}
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
	}

	structured := m.WithStructuredOutput(schema).(*ChatModel)
	schema["properties"].(map[string]any)["name"].(map[string]any)["type"] = "integer"

	got := structured.structuredSchema["properties"].(map[string]any)["name"].(map[string]any)["type"]
	if got != "string" {
		t.Fatalf("expected cloned schema to remain unchanged, got %v", got)
	}
}

// ---------- Embeddings ----------

func TestEmbedDocuments(t *testing.T) {
	vectors := [][]float64{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		jsonResponse(t, w, embedResponse{
			Model:      "nomic-embed-text",
			Embeddings: vectors,
		})
	}))
	defer srv.Close()

	e := NewEmbeddings(WithBaseURL(srv.URL))
	results, err := e.EmbedDocuments(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("EmbedDocuments error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(results))
	}
	if results[0][0] != 0.1 {
		t.Errorf("unexpected embedding value: %v", results[0][0])
	}
}

func TestEmbedQuery(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		jsonResponse(t, w, embedResponse{
			Model:      "nomic-embed-text",
			Embeddings: [][]float64{{0.1, 0.2, 0.3}},
		})
	}))
	defer srv.Close()

	e := NewEmbeddings(WithBaseURL(srv.URL))
	result, err := e.EmbedQuery(context.Background(), "hello")
	if err != nil {
		t.Fatalf("EmbedQuery error: %v", err)
	}
	if len(result) != 3 {
		t.Errorf("expected 3-dim vector, got %d", len(result))
	}
}

func TestEmbedQuery_EmptyResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		jsonResponse(t, w, embedResponse{
			Model:      "nomic-embed-text",
			Embeddings: [][]float64{},
		})
	}))
	defer srv.Close()

	e := NewEmbeddings(WithBaseURL(srv.URL))
	_, err := e.EmbedQuery(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error for empty embeddings")
	}
}

// ---------- interface check ----------

func TestChatModelImplementsInterface(t *testing.T) {
	var _ llms.ChatModel = (*ChatModel)(nil)
}
