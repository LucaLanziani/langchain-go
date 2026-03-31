package loaders

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/textsplitters"
)

// URLLoaderOption configures a URLLoader.
type URLLoaderOption func(*URLLoader)

// WithTimeout sets the HTTP client timeout.
func WithTimeout(d time.Duration) URLLoaderOption {
	return func(u *URLLoader) {
		u.client.Timeout = d
	}
}

// WithHeader adds a custom HTTP request header.
func WithHeader(key, value string) URLLoaderOption {
	return func(u *URLLoader) {
		u.headers[key] = value
	}
}

// URLLoader fetches content from a URL and returns it as a [core.Document].
// The content-type header is used to select the appropriate parser (HTML or plain text).
type URLLoader struct {
	baseLoader
	url     string
	client  http.Client
	headers map[string]string
}

// NewURLLoader creates a URLLoader for the given URL.
func NewURLLoader(url string, opts ...URLLoaderOption) *URLLoader {
	l := &URLLoader{
		url:     url,
		client:  http.Client{Timeout: 30 * time.Second},
		headers: make(map[string]string),
	}
	for _, opt := range opts {
		opt(l)
	}
	l.baseLoader.loader = l
	return l
}

// Load fetches the URL and returns a document. The parser is chosen based on
// the response Content-Type header.
func (l *URLLoader) Load(ctx context.Context) ([]*core.Document, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, l.url, nil)
	if err != nil {
		return nil, fmt.Errorf("loaders: url: create request: %w", err)
	}
	for k, v := range l.headers {
		req.Header.Set(k, v)
	}

	resp, err := l.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("loaders: url: fetch %s: %w", l.url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("loaders: url: unexpected status %d for %s", resp.StatusCode, l.url)
	}

	contentType := resp.Header.Get("Content-Type")
	meta := map[string]any{
		"source":       l.url,
		"content-type": contentType,
	}

	var pageContent string
	if strings.Contains(contentType, "text/html") {
		pageContent, _, err = parseHTML(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("loaders: url: parse html: %w", err)
		}
	} else {
		body, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return nil, fmt.Errorf("loaders: url: read body: %w", readErr)
		}
		pageContent = string(body)
	}

	return []*core.Document{core.NewDocument(pageContent, meta)}, nil
}

// LoadAndSplit fetches the URL and splits the resulting document with splitter.
func (l *URLLoader) LoadAndSplit(ctx context.Context, splitter textsplitters.TextSplitter) ([]*core.Document, error) {
	return l.baseLoader.LoadAndSplit(ctx, splitter)
}

// Ensure URLLoader implements DocumentLoader.
var _ DocumentLoader = (*URLLoader)(nil)
