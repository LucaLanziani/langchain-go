// Package retrievers provides the retriever interface and implementations
// for document retrieval.
package retrievers

import (
	"context"
	"fmt"

	"github.com/langchain-go/langchain-go/core"
	"github.com/langchain-go/langchain-go/vectorstores"
)

// Retriever is the interface for document retrievers.
// A retriever takes a query and returns relevant documents.
// It implements Runnable[string, []*core.Document].
type Retriever interface {
	core.Runnable[string, []*core.Document]

	// GetRelevantDocuments retrieves documents relevant to the query.
	GetRelevantDocuments(ctx context.Context, query string) ([]*core.Document, error)
}

// VectorStoreRetriever wraps a VectorStore as a Retriever.
type VectorStoreRetriever struct {
	store     vectorstores.VectorStore
	k         int
	name      string
}

// NewVectorStoreRetriever creates a retriever from a vector store.
func NewVectorStoreRetriever(store vectorstores.VectorStore, k int) *VectorStoreRetriever {
	if k <= 0 {
		k = 4
	}
	return &VectorStoreRetriever{
		store: store,
		k:     k,
	}
}

// WithName sets the name for tracing.
func (r *VectorStoreRetriever) WithName(name string) *VectorStoreRetriever {
	r.name = name
	return r
}

// GetName returns the retriever name.
func (r *VectorStoreRetriever) GetName() string {
	if r.name != "" {
		return r.name
	}
	return "VectorStoreRetriever"
}

// GetRelevantDocuments searches the vector store for relevant documents.
func (r *VectorStoreRetriever) GetRelevantDocuments(ctx context.Context, query string) ([]*core.Document, error) {
	return r.store.SimilaritySearch(ctx, query, r.k)
}

// Invoke retrieves documents for the given query.
func (r *VectorStoreRetriever) Invoke(ctx context.Context, input string, opts ...core.Option) ([]*core.Document, error) {
	return r.GetRelevantDocuments(ctx, input)
}

// Stream returns a single-chunk stream of retrieved documents.
func (r *VectorStoreRetriever) Stream(ctx context.Context, input string, opts ...core.Option) (*core.StreamIterator[[]*core.Document], error) {
	docs, err := r.GetRelevantDocuments(ctx, input)
	if err != nil {
		return nil, err
	}
	ch := make(chan core.StreamChunk[[]*core.Document], 1)
	ch <- core.StreamChunk[[]*core.Document]{Value: docs}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

// Batch retrieves documents for multiple queries.
func (r *VectorStoreRetriever) Batch(ctx context.Context, inputs []string, opts ...core.Option) ([][]*core.Document, error) {
	results := make([][]*core.Document, len(inputs))
	for i, input := range inputs {
		docs, err := r.GetRelevantDocuments(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", i, err)
		}
		results[i] = docs
	}
	return results, nil
}

// Ensure VectorStoreRetriever implements Retriever.
var _ Retriever = (*VectorStoreRetriever)(nil)
