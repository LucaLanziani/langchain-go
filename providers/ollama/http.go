package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// doPost marshals body as JSON, POSTs to url, reads and returns the response body.
// Returns an error if the status is not 200.
func doPost(ctx context.Context, c *http.Client, url string, body any) ([]byte, error) {
	resp, err := doRawPost(ctx, c, url, body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("ollama: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama: status %d: %s", resp.StatusCode, data)
	}
	return data, nil
}

// doRawPost marshals body as JSON and POSTs to url, returning the raw *http.Response.
// The caller is responsible for closing resp.Body.
func doRawPost(ctx context.Context, c *http.Client, url string, body any) (*http.Response, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("ollama: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("ollama: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama: request: %w", err)
	}
	return resp, nil
}
