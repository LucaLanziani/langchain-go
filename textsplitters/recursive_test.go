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

func TestWithSeparators(t *testing.T) {
	splitter := NewRecursiveCharacterTextSplitter(20, 0)
	splitter = splitter.WithSeparators([]string{"|", ""})
	if len(splitter.Separators) != 2 || splitter.Separators[0] != "|" {
		t.Errorf("expected custom separators, got %v", splitter.Separators)
	}
	text := "part1|part2|part3"
	chunks := splitter.SplitText(text)
	if len(chunks) == 0 {
		t.Fatal("expected chunks with custom separator")
	}
}

func TestSplitTextCharacterFallback(t *testing.T) {
	// Force character-by-character split by using tiny chunk size.
	splitter := NewRecursiveCharacterTextSplitter(1, 0)
	chunks := splitter.SplitText("abc")
	if len(chunks) == 0 {
		t.Fatal("expected split characters")
	}
}

func TestSplitDocumentsNilMetadata(t *testing.T) {
	splitter := NewRecursiveCharacterTextSplitter(1000, 0)
	docs := []*core.Document{
		{PageContent: "doc with no metadata"},
	}
	result := splitter.SplitDocuments(docs)
	if len(result) != 1 {
		t.Errorf("expected 1 chunk, got %d", len(result))
	}
	if result[0].Metadata != nil {
		t.Error("expected nil metadata to remain nil")
	}
}

func TestMergeSplitsOverlap(t *testing.T) {
	splitter := NewRecursiveCharacterTextSplitter(15, 5)
	// Text with multiple newline-separated pieces to trigger overlap logic.
	text := "aaa\nbbb\nccc\nddd\neee\nfff"
	chunks := splitter.SplitText(text)
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
}

func TestMergeSplitsWhitespaceChunk(t *testing.T) {
	// Trigger the whitespace trim path in mergeSplits.
	splitter := NewRecursiveCharacterTextSplitter(5, 0)
	splitter.Separators = []string{"\n", ""}
	// Text with a pure-whitespace line between content.
	text := "abc\n   \ndef"
	chunks := splitter.SplitText(text)
	// Just ensure no panic and at least one chunk.
	_ = chunks
}
