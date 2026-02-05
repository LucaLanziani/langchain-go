package core

// Document represents a piece of text with associated metadata.
// Documents are the fundamental unit of data in LangChain for
// retrieval, indexing, and processing.
type Document struct {
	// PageContent is the text content of the document.
	PageContent string `json:"page_content"`

	// Metadata contains arbitrary key-value pairs associated with the document.
	Metadata map[string]any `json:"metadata,omitempty"`

	// ID is an optional unique identifier for the document.
	ID string `json:"id,omitempty"`
}

// NewDocument creates a new Document with the given content and optional metadata.
func NewDocument(pageContent string, metadata ...map[string]any) *Document {
	doc := &Document{
		PageContent: pageContent,
	}
	if len(metadata) > 0 && metadata[0] != nil {
		doc.Metadata = metadata[0]
	}
	return doc
}
