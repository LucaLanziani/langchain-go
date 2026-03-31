//go:build pdf

package loaders

import (
	"context"
	"fmt"

	"github.com/ledongthuc/pdf"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/textsplitters"
)

// PDFLoader loads a PDF file and returns one [core.Document] per page.
// Each document's metadata includes the source path and the page number (1-based).
//
// This loader is gated behind the "pdf" build tag to keep the default module
// dependency-light. Build with -tags pdf to enable it.
type PDFLoader struct {
	baseLoader
	path string
}

// NewPDFLoader creates a PDFLoader that reads the PDF at path.
func NewPDFLoader(path string) *PDFLoader {
	l := &PDFLoader{path: path}
	l.baseLoader.loader = l
	return l
}

// Load parses the PDF and returns one document per page.
func (l *PDFLoader) Load(ctx context.Context) ([]*core.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	f, r, err := pdf.Open(l.path)
	if err != nil {
		return nil, fmt.Errorf("loaders: pdf: open %s: %w", l.path, err)
	}
	defer f.Close()

	numPages := r.NumPage()
	docs := make([]*core.Document, 0, numPages)
	for i := 1; i <= numPages; i++ {
		if err := ctx.Err(); err != nil {
			return docs, err
		}
		page := r.Page(i)
		if page.V.IsNull() {
			continue
		}
		content, err := page.GetPlainText(nil)
		if err != nil {
			return docs, fmt.Errorf("loaders: pdf: page %d of %s: %w", i, l.path, err)
		}
		doc := core.NewDocument(content, map[string]any{
			"source": l.path,
			"page":   i,
		})
		docs = append(docs, doc)
	}
	return docs, nil
}

// LoadAndSplit loads the PDF and splits the resulting documents with splitter.
func (l *PDFLoader) LoadAndSplit(ctx context.Context, splitter textsplitters.TextSplitter) ([]*core.Document, error) {
	return l.baseLoader.LoadAndSplit(ctx, splitter)
}

// Ensure PDFLoader implements DocumentLoader.
var _ DocumentLoader = (*PDFLoader)(nil)
