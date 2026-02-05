package textsplitters

import (
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
)

func TestRecursiveCharacterTextSplitter(t *testing.T) {
	splitter := NewRecursiveCharacterTextSplitter(50, 10)

	text := "Hello world. This is a test document. It has several sentences. We want to split it into chunks."
	chunks := splitter.SplitText(text)

	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}

	for i, chunk := range chunks {
		if len(chunk) > 60 { // some tolerance for overlap
			t.Errorf("chunk %d too long: %d chars: %q", i, len(chunk), chunk)
		}
	}
}

func TestRecursiveCharacterTextSplitterSmallText(t *testing.T) {
	splitter := NewRecursiveCharacterTextSplitter(1000, 0)

	text := "Short text."
	chunks := splitter.SplitText(text)

	if len(chunks) != 1 {
		t.Errorf("expected 1 chunk for short text, got %d", len(chunks))
	}
	if chunks[0] != "Short text." {
		t.Errorf("expected 'Short text.', got %q", chunks[0])
	}
}

func TestRecursiveCharacterTextSplitterParagraphs(t *testing.T) {
	splitter := NewRecursiveCharacterTextSplitter(100, 0)

	text := "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
	chunks := splitter.SplitText(text)

	if len(chunks) < 1 {
		t.Fatal("expected at least 1 chunk")
	}
}

func TestSplitDocuments(t *testing.T) {
	splitter := NewRecursiveCharacterTextSplitter(20, 0)

	docs := []*core.Document{
		{PageContent: "This is a long document that should be split.", Metadata: map[string]any{"source": "test"}},
	}

	result := splitter.SplitDocuments(docs)
	if len(result) < 2 {
		t.Errorf("expected multiple chunks, got %d", len(result))
	}

	// Check metadata is preserved.
	for _, doc := range result {
		if doc.Metadata["source"] != "test" {
			t.Error("metadata not preserved")
		}
	}
}
