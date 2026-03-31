package loaders

import (
	"context"
	"os"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/textsplitters"
)

// MarkdownLoader loads a Markdown file into a single [core.Document].
// Metadata includes source path and format "markdown".
type MarkdownLoader struct {
	baseLoader
	path string
}

// NewMarkdownLoader creates a MarkdownLoader that reads the file at path.
func NewMarkdownLoader(path string) *MarkdownLoader {
	l := &MarkdownLoader{path: path}
	l.baseLoader.loader = l
	return l
}

// Load reads the Markdown file and returns a single document.
func (l *MarkdownLoader) Load(ctx context.Context) ([]*core.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(l.path)
	if err != nil {
		return nil, err
	}
	doc := core.NewDocument(string(data), map[string]any{
		"source": l.path,
		"format": "markdown",
	})
	return []*core.Document{doc}, nil
}

// LoadAndSplit loads the file and splits the resulting document with splitter.
func (l *MarkdownLoader) LoadAndSplit(ctx context.Context, splitter textsplitters.TextSplitter) ([]*core.Document, error) {
	return l.baseLoader.LoadAndSplit(ctx, splitter)
}

// Ensure MarkdownLoader implements DocumentLoader.
var _ DocumentLoader = (*MarkdownLoader)(nil)
