// Package textsplitters provides utilities for splitting text into chunks.
package textsplitters

import (
	"strings"

	"github.com/langchain-go/langchain-go/core"
)

// RecursiveCharacterTextSplitter splits text by recursively trying different
// separators until chunks are small enough.
type RecursiveCharacterTextSplitter struct {
	// ChunkSize is the maximum size of each chunk in characters.
	ChunkSize int

	// ChunkOverlap is the number of overlapping characters between chunks.
	ChunkOverlap int

	// Separators is the list of separators to try, in order.
	Separators []string

	// LengthFunction computes the length of a string. Defaults to len().
	LengthFunction func(string) int
}

// NewRecursiveCharacterTextSplitter creates a splitter with default settings.
func NewRecursiveCharacterTextSplitter(chunkSize, chunkOverlap int) *RecursiveCharacterTextSplitter {
	return &RecursiveCharacterTextSplitter{
		ChunkSize:    chunkSize,
		ChunkOverlap: chunkOverlap,
		Separators:   []string{"\n\n", "\n", " ", ""},
		LengthFunction: func(s string) int {
			return len(s)
		},
	}
}

// WithSeparators sets custom separators.
func (s *RecursiveCharacterTextSplitter) WithSeparators(seps []string) *RecursiveCharacterTextSplitter {
	s.Separators = seps
	return s
}

// SplitText splits a text string into chunks.
func (s *RecursiveCharacterTextSplitter) SplitText(text string) []string {
	return s.splitText(text, s.Separators)
}

// SplitDocuments splits multiple documents into smaller documents.
func (s *RecursiveCharacterTextSplitter) SplitDocuments(documents []*core.Document) []*core.Document {
	var result []*core.Document
	for _, doc := range documents {
		chunks := s.SplitText(doc.PageContent)
		for _, chunk := range chunks {
			newDoc := &core.Document{
				PageContent: chunk,
				Metadata:    copyMetadata(doc.Metadata),
			}
			result = append(result, newDoc)
		}
	}
	return result
}

func (s *RecursiveCharacterTextSplitter) splitText(text string, separators []string) []string {
	// Find the appropriate separator.
	var separator string
	var newSeparators []string
	for i, sep := range separators {
		if sep == "" || strings.Contains(text, sep) {
			separator = sep
			newSeparators = separators[i+1:]
			break
		}
	}

	// Split by the chosen separator.
	var splits []string
	if separator == "" {
		// Split character by character.
		for _, ch := range text {
			splits = append(splits, string(ch))
		}
	} else {
		splits = strings.Split(text, separator)
	}

	// Merge or recursively split.
	var goodSplits []string
	var finalChunks []string

	for _, split := range splits {
		if s.LengthFunction(split) < s.ChunkSize {
			goodSplits = append(goodSplits, split)
		} else {
			if len(goodSplits) > 0 {
				finalChunks = append(finalChunks, s.mergeSplits(goodSplits, separator)...)
				goodSplits = nil
			}
			if len(newSeparators) == 0 {
				finalChunks = append(finalChunks, split)
			} else {
				finalChunks = append(finalChunks, s.splitText(split, newSeparators)...)
			}
		}
	}
	if len(goodSplits) > 0 {
		finalChunks = append(finalChunks, s.mergeSplits(goodSplits, separator)...)
	}

	return finalChunks
}

func (s *RecursiveCharacterTextSplitter) mergeSplits(splits []string, separator string) []string {
	var docs []string
	var currentDoc []string
	total := 0

	for _, d := range splits {
		dLen := s.LengthFunction(d)
		sepLen := 0
		if separator != "" {
			sepLen = s.LengthFunction(separator)
		}

		if total+dLen+(sepLen*len(currentDoc)) > s.ChunkSize && len(currentDoc) > 0 {
			doc := strings.Join(currentDoc, separator)
			if strings.TrimSpace(doc) != "" {
				docs = append(docs, doc)
			}
			// Handle overlap: keep trailing elements.
			for total > s.ChunkOverlap || (total+dLen+(sepLen*len(currentDoc)) > s.ChunkSize && total > 0) {
				if len(currentDoc) == 0 {
					break
				}
				removed := currentDoc[0]
				currentDoc = currentDoc[1:]
				total -= s.LengthFunction(removed)
				if len(currentDoc) > 0 {
					total -= sepLen
				}
			}
		}

		currentDoc = append(currentDoc, d)
		total += dLen
		if len(currentDoc) > 1 {
			total += sepLen
		}
	}

	doc := strings.Join(currentDoc, separator)
	if strings.TrimSpace(doc) != "" {
		docs = append(docs, doc)
	}

	return docs
}

func copyMetadata(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	cp := make(map[string]any, len(m))
	for k, v := range m {
		cp[k] = v
	}
	return cp
}
