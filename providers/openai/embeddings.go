package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
)

// Embeddings implements the embedding interface using OpenAI's API.
type Embeddings struct {
	opts   *Options
	model  string
}

// NewEmbeddings creates a new OpenAI Embeddings instance.
func NewEmbeddings(optFns ...OptionFunc) *Embeddings {
	opts := DefaultOptions()
	for _, fn := range optFns {
		fn(opts)
	}
	if opts.APIKey == "" {
		opts.APIKey = os.Getenv("OPENAI_API_KEY")
	}
	return &Embeddings{
		opts:  opts,
		model: "text-embedding-3-small",
	}
}

// WithEmbeddingModel sets the embedding model name.
func (e *Embeddings) WithEmbeddingModel(model string) *Embeddings {
	e.model = model
	return e
}

// EmbedDocuments embeds multiple texts.
func (e *Embeddings) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	reqBody := map[string]any{
		"model": e.model,
		"input": texts,
	}

	cm := &ChatModel{opts: e.opts, client: defaultHTTPClient()}
	respBody, err := cm.doRequest(ctx, "/embeddings", reqBody)
	if err != nil {
		return nil, err
	}

	var resp embeddingResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse embedding response: %w", err)
	}

	results := make([][]float64, len(resp.Data))
	for _, d := range resp.Data {
		results[d.Index] = d.Embedding
	}
	return results, nil
}

// EmbedQuery embeds a single query text.
func (e *Embeddings) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	results, err := e.EmbedDocuments(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}
	return results[0], nil
}

type embeddingResponse struct {
	Object string           `json:"object"`
	Data   []embeddingData  `json:"data"`
	Model  string           `json:"model"`
	Usage  *openAIUsage     `json:"usage,omitempty"`
}

type embeddingData struct {
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

func defaultHTTPClient() *http.Client {
	return &http.Client{}
}
