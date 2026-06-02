package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// Model describes an OpenAI model.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
	Created int64  `json:"created"`
}

type modelListResponse struct {
	Data []Model `json:"data"`
}

// ListModels fetches the models available to the configured API key.
func (m *ChatModel) ListModels(ctx context.Context) ([]Model, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, m.opts.BaseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("openai: build models request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.opts.APIKey)
	if m.opts.Organization != "" {
		req.Header.Set("OpenAI-Organization", m.opts.Organization)
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai: list models: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: read models response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openai: list models failed (status %d): %s", resp.StatusCode, string(body))
	}
	var parsed modelListResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("openai: parse models response: %w", err)
	}
	return parsed.Data, nil
}
