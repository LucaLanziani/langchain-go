package loaders

import (
	"context"
	"os"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/textsplitters"
)

// TextLoader loads a plain-text file into a single [core.Document].
// The document's metadata includes the source file path.
type TextLoader struct {
	baseLoader
	path string
}

// NewTextLoader creates a TextLoader that reads the file at path.
func NewTextLoader(path string) *TextLoader {
	l := &TextLoader{path: path}
	l.baseLoader.loader = l
	return l
}

// Load reads the file and returns a single document.
func (l *TextLoader) Load(ctx context.Context) ([]*core.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(l.path)
	if err != nil {
		return nil, err
	}
	doc := core.NewDocument(string(data), map[string]any{
		"source": l.path,
	})
	return []*core.Document{doc}, nil
}

// LoadAndSplit loads the file and splits the resulting document with splitter.
func (l *TextLoader) LoadAndSplit(ctx context.Context, splitter textsplitters.TextSplitter) ([]*core.Document, error) {
	return l.baseLoader.LoadAndSplit(ctx, splitter)
}

// Ensure TextLoader implements DocumentLoader.
var _ DocumentLoader = (*TextLoader)(nil)
