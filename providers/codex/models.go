package codex

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Model describes a model exposed by the Codex backend.
type Model struct {
	ID                       string   `json:"id"`
	Model                    string   `json:"model"`
	DisplayName              string   `json:"displayName"`
	Description              string   `json:"description"`
	Hidden                   bool     `json:"hidden"`
	IsDefault                bool     `json:"isDefault"`
	SupportedReasoningEfforts []string `json:"supportedReasoningEfforts"`
	DefaultReasoningEffort   string   `json:"defaultReasoningEffort"`
}

type modelListResponse struct {
	Data       []Model `json:"data"`
	NextCursor *string `json:"nextCursor"`
}

// ListModels fetches the models available to the authenticated ChatGPT account.
func (m *ChatModel) ListModels(ctx context.Context) ([]Model, error) {
	body, err := m.fetchModelsRaw(ctx)
	if err != nil {
		return nil, err
	}
	var parsed modelListResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("codex: parse models response: %w", err)
	}
	return parsed.Data, nil
}

// ListModelsRaw returns the raw JSON body from the /models endpoint. Useful for
// debugging when ListModels returns nothing because the schema shifted.
func (m *ChatModel) ListModelsRaw(ctx context.Context) ([]byte, error) {
	return m.fetchModelsRaw(ctx)
}

func (m *ChatModel) fetchModelsRaw(ctx context.Context) ([]byte, error) {
	accessToken, accountID, err := m.auth.AccessToken(ctx)
	if err != nil {
		return nil, err
	}

	url := strings.TrimRight(m.opts.BaseURL, "/") + "/models?client_version=" + clientVersion
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("codex: build request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+accessToken)
	if accountID != "" {
		req.Header.Set("ChatGPT-Account-ID", accountID)
	}
	req.Header.Set("originator", m.opts.Originator)
	if m.opts.UserAgent != "" {
		req.Header.Set("User-Agent", m.opts.UserAgent)
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("codex: list models: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("codex: read models response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("codex: list models failed (status %d): %s", resp.StatusCode, string(body))
	}
	return body, nil
}
