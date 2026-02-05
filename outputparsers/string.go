// Package outputparsers provides parsers for extracting structured data
// from language model outputs.
package outputparsers

import (
	"context"

	"github.com/langchain-go/langchain-go/core"
)

// StringOutputParser extracts the text content from an AI message.
// It implements Runnable[*core.AIMessage, string].
type StringOutputParser struct {
	name string
}

// NewStringOutputParser creates a new StringOutputParser.
func NewStringOutputParser() *StringOutputParser {
	return &StringOutputParser{}
}

// WithName sets the name for tracing.
func (p *StringOutputParser) WithName(name string) *StringOutputParser {
	p.name = name
	return p
}

// GetName returns the name of this parser.
func (p *StringOutputParser) GetName() string {
	if p.name != "" {
		return p.name
	}
	return "StringOutputParser"
}

// Parse extracts the string content from an AI message.
func (p *StringOutputParser) Parse(msg *core.AIMessage) (string, error) {
	return msg.GetContent(), nil
}

// ParseMessage extracts content from any Message interface.
func (p *StringOutputParser) ParseMessage(msg core.Message) (string, error) {
	return msg.GetContent(), nil
}

// Invoke parses the message.
func (p *StringOutputParser) Invoke(ctx context.Context, input *core.AIMessage, opts ...core.Option) (string, error) {
	return p.Parse(input)
}

// Stream returns a single-chunk stream of the parsed content.
func (p *StringOutputParser) Stream(ctx context.Context, input *core.AIMessage, opts ...core.Option) (*core.StreamIterator[string], error) {
	result, err := p.Parse(input)
	if err != nil {
		return nil, err
	}
	ch := make(chan core.StreamChunk[string], 1)
	ch <- core.StreamChunk[string]{Value: result}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

// Batch parses multiple messages.
func (p *StringOutputParser) Batch(ctx context.Context, inputs []*core.AIMessage, opts ...core.Option) ([]string, error) {
	results := make([]string, len(inputs))
	for i, input := range inputs {
		result, err := p.Parse(input)
		if err != nil {
			return nil, err
		}
		results[i] = result
	}
	return results, nil
}
