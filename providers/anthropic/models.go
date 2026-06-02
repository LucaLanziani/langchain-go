package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// Model describes an Anthropic model exposed by the /v1/models endpoint.
type Model struct {
	ID          string `json:"id"`
	Type        string `json:"type"`
	DisplayName string `json:"display_name"`
	CreatedAt   string `json:"created_at"`
}

type modelListResponse struct {
	Data    []Model `json:"data"`
	HasMore bool    `json:"has_more"`
	FirstID string  `json:"first_id"`
	LastID  string  `json:"last_id"`
}

// ListModels fetches the models available to the configured API key.
func (m *ChatModel) ListModels(ctx context.Context) ([]Model, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, m.opts.BaseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("anthropic: build models request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("x-api-key", m.opts.APIKey)
	req.Header.Set("anthropic-version", anthropicAPIVersion)

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic: list models: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("anthropic: read models response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("anthropic: list models failed (status %d): %s", resp.StatusCode, string(body))
	}
	var parsed modelListResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("anthropic: parse models response: %w", err)
	}
	return parsed.Data, nil
}
