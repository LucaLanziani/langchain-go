package codex

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
)

func TestParseJWTClaims_ExtractsAccountID(t *testing.T) {
	idToken := buildFakeIDToken(t, time.Now().Add(time.Hour), "acct-123")
	claims, ok := parseJWTClaims(idToken)
	if !ok {
		t.Fatal("expected to parse JWT claims")
	}
	if claims.Auth.ChatGPTAccountID != "acct-123" {
		t.Fatalf("got account id %q, want acct-123", claims.Auth.ChatGPTAccountID)
	}
}

func TestNeedsRefresh(t *testing.T) {
	cases := map[string]struct {
		expiresIn time.Duration
		want      bool
	}{
		"expired":   {-time.Minute, true},
		"near":      {2 * time.Minute, true},
		"plentiful": {time.Hour, false},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			token := buildFakeIDToken(t, time.Now().Add(tc.expiresIn), "")
			if got := needsRefresh(token); got != tc.want {
				t.Fatalf("needsRefresh = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestAuthManager_RefreshesExpiredToken(t *testing.T) {
	dir := t.TempDir()
	authPath := filepath.Join(dir, "auth.json")

	expiredAccess := buildFakeIDToken(t, time.Now().Add(-time.Minute), "")
	freshAccess := buildFakeIDToken(t, time.Now().Add(time.Hour), "")
	freshID := buildFakeIDToken(t, time.Now().Add(time.Hour), "acct-new")

	writeAuthFile(t, authPath, &AuthFile{
		Tokens: &AuthToken{
			AccessToken:  expiredAccess,
			RefreshToken: "rt-original",
			IDToken:      buildFakeIDToken(t, time.Now().Add(time.Hour), "acct-old"),
			AccountID:    "acct-old",
		},
	})

	var capturedBody map[string]string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"id_token":      freshID,
			"access_token":  freshAccess,
			"refresh_token": "rt-rotated",
		})
	}))
	defer server.Close()

	mgr, err := NewAuthManager(authPath)
	if err != nil {
		t.Fatal(err)
	}
	// Redirect refresh traffic at the fake server by swapping the http client and URL.
	mgr.httpClient = server.Client()
	prevURL := authTokenURL
	authTokenURL = server.URL
	t.Cleanup(func() { authTokenURL = prevURL })

	token, accountID, err := mgr.AccessToken(context.Background())
	if err != nil {
		t.Fatalf("AccessToken: %v", err)
	}
	if token != freshAccess {
		t.Fatalf("token not refreshed; got %q", token)
	}
	if accountID != "acct-new" {
		t.Fatalf("account id not refreshed; got %q", accountID)
	}
	if capturedBody["refresh_token"] != "rt-original" {
		t.Fatalf("server saw refresh_token %q, want rt-original", capturedBody["refresh_token"])
	}
	if capturedBody["client_id"] != authClientID {
		t.Fatalf("server saw client_id %q, want %s", capturedBody["client_id"], authClientID)
	}

	// File should have been persisted with rotated refresh token.
	reloaded, err := loadAuthFile(authPath)
	if err != nil {
		t.Fatal(err)
	}
	if reloaded.Tokens.RefreshToken != "rt-rotated" {
		t.Fatalf("refresh token not persisted, got %q", reloaded.Tokens.RefreshToken)
	}
}

func TestBuildRequestBody_ShapesResponsesAPIPayload(t *testing.T) {
	m := &ChatModel{
		opts:      DefaultOptions(),
		sessionID: "sess-1",
		boundTools: []llms.ToolDefinition{
			{Name: "read_file", Description: "Read a file", Parameters: map[string]any{"type": "object"}},
		},
	}
	m.opts.ReasoningEffort = "medium"
	m.opts.PromptCacheKey = "cache-key-1"
	m.opts.Model = "gpt-5-codex"

	msgs := []core.Message{
		core.NewSystemMessage("You are Jarvis."),
		core.NewHumanMessage("Hello"),
		core.NewAIMessage("Hi back"),
		&core.ToolMessage{
			BaseMessage: core.BaseMessage{Content: "file contents"},
			ToolCallID:  "call-42",
		},
	}

	body, err := m.buildRequestBody(msgs, core.ApplyOptions())
	if err != nil {
		t.Fatal(err)
	}
	if body["model"] != "gpt-5-codex" {
		t.Fatalf("model = %v", body["model"])
	}
	if body["instructions"] != "You are Jarvis." {
		t.Fatalf("instructions = %v", body["instructions"])
	}
	if body["stream"] != true {
		t.Fatalf("expected stream=true")
	}
	if body["store"] != false {
		t.Fatalf("expected store=false")
	}
	if body["prompt_cache_key"] != "cache-key-1" {
		t.Fatalf("prompt_cache_key = %v", body["prompt_cache_key"])
	}

	reasoning, ok := body["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning missing or wrong type: %T", body["reasoning"])
	}
	if reasoning["effort"] != "medium" {
		t.Fatalf("reasoning.effort = %v", reasoning["effort"])
	}

	items, ok := body["input"].([]map[string]any)
	if !ok {
		t.Fatalf("input type %T", body["input"])
	}
	// human, assistant message, tool result — 3 items (system is in instructions).
	if len(items) != 3 {
		t.Fatalf("input items = %d, want 3 (got %+v)", len(items), items)
	}
	if items[0]["role"] != "user" {
		t.Fatalf("first item role = %v", items[0]["role"])
	}
	if items[2]["type"] != "function_call_output" || items[2]["call_id"] != "call-42" {
		t.Fatalf("tool result item shape wrong: %+v", items[2])
	}

	tools, ok := body["tools"].([]map[string]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools shape wrong: %+v", body["tools"])
	}
	if tools[0]["name"] != "read_file" || tools[0]["type"] != "function" {
		t.Fatalf("tool entry wrong: %+v", tools[0])
	}
}

func TestDecodeSSE_StreamsContentToolCallAndUsage(t *testing.T) {
	events := strings.Join([]string{
		"event: response.output_text.delta",
		`data: {"type":"response.output_text.delta","delta":"Hello "}`,
		"",
		`data: {"type":"response.output_text.delta","delta":"world"}`,
		"",
		`data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"c1","name":"do_thing","arguments":"{\"x\":1}"}}`,
		"",
		`data: {"type":"response.completed","response":{"id":"resp-1","usage":{"input_tokens":12,"output_tokens":7,"total_tokens":19}}}`,
		"",
	}, "\n")

	ch := make(chan core.StreamChunk[*core.AIMessage], 8)
	go func() {
		decodeSSE(context.Background(), strings.NewReader(events), ch)
		close(ch)
	}()

	var (
		text       strings.Builder
		toolCalls  []core.ToolCall
		usageSeen  bool
		finishSeen bool
	)
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("unexpected stream error: %v", chunk.Err)
		}
		text.WriteString(chunk.Value.GetContent())
		toolCalls = append(toolCalls, chunk.Value.ToolCalls...)
		if chunk.Value.UsageMetadata != nil {
			usageSeen = true
			if chunk.Value.UsageMetadata.TotalTokens != 19 {
				t.Fatalf("total tokens = %d", chunk.Value.UsageMetadata.TotalTokens)
			}
		}
		if chunk.Value.ResponseMetadata != nil {
			if fr, _ := chunk.Value.ResponseMetadata["finish_reason"].(string); fr == "stop" {
				finishSeen = true
			}
		}
	}

	if text.String() != "Hello world" {
		t.Fatalf("text = %q", text.String())
	}
	if len(toolCalls) != 1 || toolCalls[0].Name != "do_thing" || string(toolCalls[0].Args) != `{"x":1}` {
		t.Fatalf("tool calls wrong: %+v", toolCalls)
	}
	if !usageSeen {
		t.Fatal("expected usage to be emitted on response.completed")
	}
	if !finishSeen {
		t.Fatal("expected finish_reason=stop on response.completed")
	}
}

// helpers

func buildFakeIDToken(t *testing.T, exp time.Time, accountID string) string {
	t.Helper()
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none","typ":"JWT"}`))
	payload := map[string]any{
		"exp": exp.Unix(),
		"https://api.openai.com/auth": map[string]any{
			"chatgpt_account_id": accountID,
		},
	}
	payloadJSON, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	body := base64.RawURLEncoding.EncodeToString(payloadJSON)
	sig := base64.RawURLEncoding.EncodeToString([]byte("sig"))
	return fmt.Sprintf("%s.%s.%s", header, body, sig)
}

func writeAuthFile(t *testing.T, path string, af *AuthFile) {
	t.Helper()
	if err := saveAuthFile(path, af); err != nil {
		t.Fatal(err)
	}
}
