// Package loaders provides document loaders for ingesting content from
// various sources (files, URLs, directories) into [core.Document] slices.
package loaders

import (
	"context"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/textsplitters"
)

// DocumentLoader is the interface implemented by all loaders.
// Load reads content from the source and returns a slice of documents.
type DocumentLoader interface {
	Load(ctx context.Context) ([]*core.Document, error)
}

// baseLoader provides a generic LoadAndSplit implementation that can be
// embedded by concrete loaders.
type baseLoader struct {
	loader DocumentLoader
}

// LoadAndSplit loads documents from the underlying loader and then splits
// them using the provided TextSplitter.
func (b *baseLoader) LoadAndSplit(ctx context.Context, splitter textsplitters.TextSplitter) ([]*core.Document, error) {
	docs, err := b.loader.Load(ctx)
	if err != nil {
		return nil, err
	}
	return splitter.SplitDocuments(docs), nil
}
