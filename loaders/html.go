package loaders

import (
	"bytes"
	"context"
	"io"
	"os"
	"strings"

	"golang.org/x/net/html"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/textsplitters"
)

// HTMLLoader loads an HTML file into a single [core.Document].
// It strips tags, extracts the page title, and sets source/title metadata.
type HTMLLoader struct {
	baseLoader
	path string
}

// NewHTMLLoader creates an HTMLLoader that reads the file at path.
func NewHTMLLoader(path string) *HTMLLoader {
	l := &HTMLLoader{path: path}
	l.baseLoader.loader = l
	return l
}

// Load reads and parses the HTML file, returning a single document whose
// content is the visible text with tags stripped.
func (l *HTMLLoader) Load(ctx context.Context) ([]*core.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(l.path)
	if err != nil {
		return nil, err
	}
	text, title, err := parseHTML(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	meta := map[string]any{
		"source": l.path,
	}
	if title != "" {
		meta["title"] = title
	}
	doc := core.NewDocument(text, meta)
	return []*core.Document{doc}, nil
}

// LoadAndSplit loads the file and splits the resulting document with splitter.
func (l *HTMLLoader) LoadAndSplit(ctx context.Context, splitter textsplitters.TextSplitter) ([]*core.Document, error) {
	return l.baseLoader.LoadAndSplit(ctx, splitter)
}

// parseHTML extracts visible text content and the page title from an HTML reader.
func parseHTML(r io.Reader) (text, title string, err error) {
	node, err := html.Parse(r)
	if err != nil {
		return "", "", err
	}

	var textBuf strings.Builder
	var titleBuf strings.Builder

	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			switch n.Data {
			case "script", "style":
				// Skip these subtrees entirely.
				return
			case "title":
				// Collect text inside <title> into titleBuf.
				for c := n.FirstChild; c != nil; c = c.NextSibling {
					if c.Type == html.TextNode {
						titleBuf.WriteString(c.Data)
					}
				}
				return
			}
		}
		if n.Type == html.TextNode {
			trimmed := strings.TrimSpace(n.Data)
			if trimmed != "" {
				if textBuf.Len() > 0 {
					textBuf.WriteByte('\n')
				}
				textBuf.WriteString(trimmed)
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(node)
	return textBuf.String(), strings.TrimSpace(titleBuf.String()), nil
}

// Ensure HTMLLoader implements DocumentLoader.
var _ DocumentLoader = (*HTMLLoader)(nil)
