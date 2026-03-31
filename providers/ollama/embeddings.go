package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// Embeddings implements text embeddings using Ollama's /api/embed endpoint.
type Embeddings struct {
	opts   *options
	client *http.Client
}

// NewEmbeddings creates a new Ollama Embeddings instance.
func NewEmbeddings(optFns ...OptionFunc) *Embeddings {
	o := defaultOptions()
	// Default embedding model
	o.Model = "nomic-embed-text"
	for _, fn := range optFns {
		fn(o)
	}
	return &Embeddings{
		opts:   o,
		client: &http.Client{},
	}
}

// EmbedDocuments embeds multiple texts using Ollama's batch embedding endpoint.
func (e *Embeddings) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	req := &embedRequest{
		Model:     e.opts.Model,
		Input:     texts,
		KeepAlive: e.opts.KeepAlive,
	}

	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("ollama: failed to marshal embed request: %w", err)
	}

	cm := &ChatModel{opts: e.opts, client: e.client}
	respBody, err := cm.doRequest(ctx, "/api/embed", json.RawMessage(reqJSON))
	if err != nil {
		return nil, err
	}

	var resp embedResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, fmt.Errorf("ollama: failed to parse embed response: %w", err)
	}

	return resp.Embeddings, nil
}

// EmbedQuery embeds a single text.
func (e *Embeddings) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	results, err := e.EmbedDocuments(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("ollama: no embedding returned")
	}
	return results[0], nil
}
