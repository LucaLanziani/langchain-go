package outputparsers

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/LucaLanziani/langchain-go/core"
)

// jsonBlockRegex matches ```json ... ``` code blocks.
var jsonBlockRegex = regexp.MustCompile("(?s)```(?:json)?\\s*\n?(.*?)\\s*```")

// JSONOutputParser parses JSON from LLM output into a Go value.
// It implements Runnable[*core.AIMessage, T].
type JSONOutputParser[T any] struct {
	name string
}

// NewJSONOutputParser creates a new JSONOutputParser for the given type.
func NewJSONOutputParser[T any]() *JSONOutputParser[T] {
	return &JSONOutputParser[T]{}
}

// WithName sets the name for tracing.
func (p *JSONOutputParser[T]) WithName(name string) *JSONOutputParser[T] {
	p.name = name
	return p
}

// GetName returns the name of this parser.
func (p *JSONOutputParser[T]) GetName() string {
	if p.name != "" {
		return p.name
	}
	return "JSONOutputParser"
}

// GetFormatInstructions returns instructions for the model on how to format output.
func (p *JSONOutputParser[T]) GetFormatInstructions() string {
	return `Return a JSON object. If you use a code block, use the json language tag.`
}

// Parse extracts and parses JSON from the AI message content.
func (p *JSONOutputParser[T]) Parse(msg *core.AIMessage) (T, error) {
	return p.ParseString(msg.GetContent())
}

// ParseMessage extracts and parses JSON from any Message interface.
func (p *JSONOutputParser[T]) ParseMessage(msg core.Message) (T, error) {
	return p.ParseString(msg.GetContent())
}

// ParseString parses JSON from a raw string, handling code blocks.
func (p *JSONOutputParser[T]) ParseString(text string) (T, error) {
	var result T

	// Try to extract from ```json ... ``` code block first.
	if matches := jsonBlockRegex.FindStringSubmatch(text); len(matches) > 1 {
		text = matches[1]
	} else {
		// Try to find JSON object or array in the text.
		text = strings.TrimSpace(text)
	}

	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return result, fmt.Errorf("failed to parse JSON output: %w\nRaw text: %s", err, text)
	}
	return result, nil
}

// Invoke parses the message.
func (p *JSONOutputParser[T]) Invoke(ctx context.Context, input *core.AIMessage, opts ...core.Option) (T, error) {
	return p.Parse(input)
}

// Stream returns a single-chunk stream of the parsed result.
func (p *JSONOutputParser[T]) Stream(ctx context.Context, input *core.AIMessage, opts ...core.Option) (*core.StreamIterator[T], error) {
	result, err := p.Parse(input)
	if err != nil {
		return nil, err
	}
	ch := make(chan core.StreamChunk[T], 1)
	ch <- core.StreamChunk[T]{Value: result}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

// Batch parses multiple messages.
func (p *JSONOutputParser[T]) Batch(ctx context.Context, inputs []*core.AIMessage, opts ...core.Option) ([]T, error) {
	results := make([]T, len(inputs))
	for i, input := range inputs {
		result, err := p.Parse(input)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}
