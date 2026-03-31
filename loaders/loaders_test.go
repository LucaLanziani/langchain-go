package loaders

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/LucaLanziani/langchain-go/textsplitters"
)

// ---------------------------------------------------------------------------
// TextLoader
// ---------------------------------------------------------------------------

func TestTextLoader_Load(t *testing.T) {
	loader := NewTextLoader("testdata/sample.txt")
	docs, err := loader.Load(context.Background())
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	if !strings.Contains(docs[0].PageContent, "Hello, world!") {
		t.Errorf("unexpected content: %q", docs[0].PageContent)
	}
	if docs[0].Metadata["source"] != "testdata/sample.txt" {
		t.Errorf("unexpected source metadata: %v", docs[0].Metadata["source"])
	}
}

func TestTextLoader_LoadMissingFile(t *testing.T) {
	loader := NewTextLoader("testdata/nonexistent.txt")
	_, err := loader.Load(context.Background())
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestTextLoader_ContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	loader := NewTextLoader("testdata/sample.txt")
	_, err := loader.Load(ctx)
	if err == nil {
		t.Fatal("expected context error")
	}
}

func TestTextLoader_LoadAndSplit(t *testing.T) {
	loader := NewTextLoader("testdata/sample.txt")
	splitter := textsplitters.NewRecursiveCharacterTextSplitter(20, 0)
	docs, err := loader.LoadAndSplit(context.Background(), splitter)
	if err != nil {
		t.Fatalf("LoadAndSplit failed: %v", err)
	}
	if len(docs) == 0 {
		t.Fatal("expected at least one chunk")
	}
	// All chunks should inherit source metadata.
	for _, doc := range docs {
		if doc.Metadata["source"] != "testdata/sample.txt" {
			t.Errorf("metadata not propagated, got: %v", doc.Metadata)
		}
	}
}

// ---------------------------------------------------------------------------
// MarkdownLoader
// ---------------------------------------------------------------------------

func TestMarkdownLoader_Load(t *testing.T) {
	loader := NewMarkdownLoader("testdata/sample.md")
	docs, err := loader.Load(context.Background())
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	if docs[0].Metadata["format"] != "markdown" {
		t.Errorf("expected format=markdown, got %v", docs[0].Metadata["format"])
	}
	if !strings.Contains(docs[0].PageContent, "Sample Markdown") {
		t.Errorf("unexpected content: %q", docs[0].PageContent)
	}
}

// ---------------------------------------------------------------------------
// HTMLLoader
// ---------------------------------------------------------------------------

func TestHTMLLoader_Load(t *testing.T) {
	loader := NewHTMLLoader("testdata/sample.html")
	docs, err := loader.Load(context.Background())
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	content := docs[0].PageContent
	if !strings.Contains(content, "Hello HTML") {
		t.Errorf("expected body text, got: %q", content)
	}
	// Script and style content must be stripped.
	if strings.Contains(content, "var x") {
		t.Errorf("script content should be stripped, got: %q", content)
	}
	if strings.Contains(content, "margin") {
		t.Errorf("style content should be stripped, got: %q", content)
	}
	if docs[0].Metadata["title"] != "Sample Page" {
		t.Errorf("expected title=Sample Page, got %v", docs[0].Metadata["title"])
	}
}

func TestHTMLLoader_LoadMissingFile(t *testing.T) {
	loader := NewHTMLLoader("testdata/nonexistent.html")
	_, err := loader.Load(context.Background())
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

// ---------------------------------------------------------------------------
// URLLoader
// ---------------------------------------------------------------------------

func TestURLLoader_PlainText(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		fmt.Fprint(w, "hello from server")
	}))
	defer srv.Close()

	loader := NewURLLoader(srv.URL)
	docs, err := loader.Load(context.Background())
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	if docs[0].PageContent != "hello from server" {
		t.Errorf("unexpected content: %q", docs[0].PageContent)
	}
	if docs[0].Metadata["source"] != srv.URL {
		t.Errorf("unexpected source: %v", docs[0].Metadata["source"])
	}
}

func TestURLLoader_HTML(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, "<html><head><title>Test</title></head><body><p>Hello HTML</p></body></html>")
	}))
	defer srv.Close()

	loader := NewURLLoader(srv.URL)
	docs, err := loader.Load(context.Background())
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	if !strings.Contains(docs[0].PageContent, "Hello HTML") {
		t.Errorf("unexpected content: %q", docs[0].PageContent)
	}
}

func TestURLLoader_NonSuccessStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer srv.Close()

	loader := NewURLLoader(srv.URL)
	_, err := loader.Load(context.Background())
	if err == nil {
		t.Fatal("expected error for non-success status")
	}
}

func TestURLLoader_Timeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Block until the request context is cancelled.
		<-r.Context().Done()
	}))
	defer srv.Close()

	loader := NewURLLoader(srv.URL, WithTimeout(50*time.Millisecond))
	_, err := loader.Load(context.Background())
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

func TestURLLoader_CustomHeader(t *testing.T) {
	var received string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		received = r.Header.Get("X-Custom")
		w.Header().Set("Content-Type", "text/plain")
		fmt.Fprint(w, "ok")
	}))
	defer srv.Close()

	loader := NewURLLoader(srv.URL, WithHeader("X-Custom", "value123"))
	_, err := loader.Load(context.Background())
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if received != "value123" {
		t.Errorf("expected header value123, got %q", received)
	}
}

// ---------------------------------------------------------------------------
// DirectoryLoader
// ---------------------------------------------------------------------------

func TestDirectoryLoader_Load(t *testing.T) {
	loader := NewDirectoryLoader("testdata/")
	docs, err := loader.Load(context.Background())
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if len(docs) == 0 {
		t.Fatal("expected at least one document")
	}
	// Verify sources are set.
	for _, doc := range docs {
		if doc.Metadata["source"] == "" {
			t.Errorf("missing source metadata on doc: %q", doc.PageContent[:min(30, len(doc.PageContent))])
		}
	}
}

func TestDirectoryLoader_NonRecursive(t *testing.T) {
	// Create a temp dir with a subdirectory containing a file.
	dir := t.TempDir()
	sub := filepath.Join(dir, "sub")
	if err := os.Mkdir(sub, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "top.txt"), []byte("top"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sub, "nested.txt"), []byte("nested"), 0644); err != nil {
		t.Fatal(err)
	}

	loader := NewDirectoryLoader(dir, WithRecursive(false))
	docs, err := loader.Load(context.Background())
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if len(docs) != 1 {
		t.Errorf("expected 1 doc (non-recursive), got %d", len(docs))
	}
}

func TestDirectoryLoader_GlobFilter(t *testing.T) {
	loader := NewDirectoryLoader("testdata/", WithGlob("*.txt"))
	docs, err := loader.Load(context.Background())
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	for _, doc := range docs {
		src := fmt.Sprintf("%v", doc.Metadata["source"])
		if !strings.HasSuffix(src, ".txt") {
			t.Errorf("glob filter failed: found non-txt file %q", src)
		}
	}
}

func TestDirectoryLoader_ContextCancelled(t *testing.T) {
	dir := t.TempDir()
	for i := 0; i < 5; i++ {
		name := filepath.Join(dir, fmt.Sprintf("file%d.txt", i))
		if err := os.WriteFile(name, []byte("content"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	loader := NewDirectoryLoader(dir)
	_, err := loader.Load(ctx)
	// Either context error or partial load — we just want no panic.
	_ = err
}

func TestDirectoryLoader_CustomLoaderMapping(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "custom.xyz")
	if err := os.WriteFile(path, []byte("custom content"), 0644); err != nil {
		t.Fatal(err)
	}

	loader := NewDirectoryLoader(dir,
		WithLoaderMapping(".xyz", func(p string) DocumentLoader {
			return NewTextLoader(p)
		}),
	)
	docs, err := loader.Load(context.Background())
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	if docs[0].PageContent != "custom content" {
		t.Errorf("unexpected content: %q", docs[0].PageContent)
	}
}

func TestDirectoryLoader_MissingDir(t *testing.T) {
	loader := NewDirectoryLoader("/nonexistent/path/that/does/not/exist")
	_, err := loader.Load(context.Background())
	if err == nil {
		t.Fatal("expected error for missing directory")
	}
}

func TestMultiError(t *testing.T) {
	me := MultiError{fmt.Errorf("err1"), fmt.Errorf("err2")}
	msg := me.Error()
	if !strings.Contains(msg, "err1") || !strings.Contains(msg, "err2") {
		t.Errorf("unexpected MultiError message: %q", msg)
	}
}

func TestMultiError_Single(t *testing.T) {
	me := MultiError{fmt.Errorf("only error")}
	if me.Error() != "only error" {
		t.Errorf("unexpected single error: %q", me.Error())
	}
}

// ---------------------------------------------------------------------------
// parseHTML
// ---------------------------------------------------------------------------

func TestParseHTML_StripsTags(t *testing.T) {
	input := `<html><body><p>Hello</p><p>World</p></body></html>`
	text, _, err := parseHTML(strings.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(text, "Hello") || !strings.Contains(text, "World") {
		t.Errorf("unexpected text: %q", text)
	}
}

func TestParseHTML_ExtractsTitle(t *testing.T) {
	input := `<html><head><title>My Title</title></head><body><p>Body</p></body></html>`
	_, title, err := parseHTML(strings.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	if title != "My Title" {
		t.Errorf("expected title 'My Title', got %q", title)
	}
}

// min returns the smaller of a and b.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
