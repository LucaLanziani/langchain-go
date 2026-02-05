// Package chains provides common chain patterns for composing LangChain components.
package chains

import (
	"context"
	"fmt"
	"strings"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
	"github.com/LucaLanziani/langchain-go/prompts"
	"github.com/LucaLanziani/langchain-go/retrievers"
)

// LLMChain is the simplest chain: prompt -> model -> output.
// It implements Runnable[map[string]any, string].
type LLMChain struct {
	prompt *prompts.ChatPromptTemplate
	llm    llms.ChatModel
	name   string
}

// NewLLMChain creates a new LLMChain.
func NewLLMChain(llm llms.ChatModel, prompt *prompts.ChatPromptTemplate) *LLMChain {
	return &LLMChain{prompt: prompt, llm: llm}
}

// GetName returns the chain name.
func (c *LLMChain) GetName() string {
	if c.name != "" {
		return c.name
	}
	return "LLMChain"
}

// Invoke runs the chain.
func (c *LLMChain) Invoke(ctx context.Context, input map[string]any, opts ...core.Option) (string, error) {
	messages, err := c.prompt.FormatMessages(input)
	if err != nil {
		return "", fmt.Errorf("prompt format error: %w", err)
	}

	response, err := c.llm.Invoke(ctx, messages, opts...)
	if err != nil {
		return "", fmt.Errorf("LLM error: %w", err)
	}

	return response.Content, nil
}

// Stream runs the chain with streaming output.
func (c *LLMChain) Stream(ctx context.Context, input map[string]any, opts ...core.Option) (*core.StreamIterator[string], error) {
	messages, err := c.prompt.FormatMessages(input)
	if err != nil {
		return nil, fmt.Errorf("prompt format error: %w", err)
	}

	stream, err := c.llm.Stream(ctx, messages, opts...)
	if err != nil {
		return nil, fmt.Errorf("LLM stream error: %w", err)
	}

	// Transform AI message chunks to strings.
	outCh := make(chan core.StreamChunk[string], 64)
	go func() {
		defer close(outCh)
		for {
			msg, ok, err := stream.Next()
			if err != nil {
				outCh <- core.StreamChunk[string]{Err: err}
				return
			}
			if !ok {
				return
			}
			outCh <- core.StreamChunk[string]{Value: msg.Content}
		}
	}()

	return core.NewStreamIterator(outCh), nil
}

// Batch runs the chain for multiple inputs.
func (c *LLMChain) Batch(ctx context.Context, inputs []map[string]any, opts ...core.Option) ([]string, error) {
	results := make([]string, len(inputs))
	for i, input := range inputs {
		result, err := c.Invoke(ctx, input, opts...)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// StuffDocumentsChain combines retrieved documents into a single context
// and passes them to an LLM chain.
type StuffDocumentsChain struct {
	llmChain    *LLMChain
	documentKey string
	inputKey    string
	separator   string
	name        string
}

// NewStuffDocumentsChain creates a chain that "stuffs" all documents into the prompt.
func NewStuffDocumentsChain(llmChain *LLMChain) *StuffDocumentsChain {
	return &StuffDocumentsChain{
		llmChain:    llmChain,
		documentKey: "context",
		inputKey:    "input_documents",
		separator:   "\n\n",
	}
}

// GetName returns the chain name.
func (c *StuffDocumentsChain) GetName() string {
	if c.name != "" {
		return c.name
	}
	return "StuffDocumentsChain"
}

// Invoke runs the chain with documents.
func (c *StuffDocumentsChain) Invoke(ctx context.Context, input map[string]any, opts ...core.Option) (string, error) {
	docsRaw, ok := input[c.inputKey]
	if !ok {
		return "", fmt.Errorf("missing input key %q", c.inputKey)
	}
	docs, ok := docsRaw.([]*core.Document)
	if !ok {
		return "", fmt.Errorf("input key %q must be []*core.Document", c.inputKey)
	}

	// Combine document contents.
	var contents []string
	for _, doc := range docs {
		contents = append(contents, doc.PageContent)
	}
	combinedContext := strings.Join(contents, c.separator)

	// Pass to LLM chain.
	mergedInput := make(map[string]any)
	for k, v := range input {
		mergedInput[k] = v
	}
	mergedInput[c.documentKey] = combinedContext

	return c.llmChain.Invoke(ctx, mergedInput, opts...)
}

// Stream streams the chain output.
func (c *StuffDocumentsChain) Stream(ctx context.Context, input map[string]any, opts ...core.Option) (*core.StreamIterator[string], error) {
	result, err := c.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	ch := make(chan core.StreamChunk[string], 1)
	ch <- core.StreamChunk[string]{Value: result}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

// Batch runs the chain for multiple inputs.
func (c *StuffDocumentsChain) Batch(ctx context.Context, inputs []map[string]any, opts ...core.Option) ([]string, error) {
	results := make([]string, len(inputs))
	for i, input := range inputs {
		result, err := c.Invoke(ctx, input, opts...)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// RetrievalQA combines a retriever with an LLM to answer questions.
// It retrieves relevant documents and uses them as context.
type RetrievalQA struct {
	retriever retrievers.Retriever
	chain     *StuffDocumentsChain
	queryKey  string
	name      string
}

// NewRetrievalQA creates a new RetrievalQA chain.
func NewRetrievalQA(retriever retrievers.Retriever, llmChain *LLMChain) *RetrievalQA {
	return &RetrievalQA{
		retriever: retriever,
		chain:     NewStuffDocumentsChain(llmChain),
		queryKey:  "query",
	}
}

// GetName returns the chain name.
func (r *RetrievalQA) GetName() string {
	if r.name != "" {
		return r.name
	}
	return "RetrievalQA"
}

// Invoke retrieves documents and answers the query.
func (r *RetrievalQA) Invoke(ctx context.Context, input map[string]any, opts ...core.Option) (string, error) {
	query, ok := input[r.queryKey]
	if !ok {
		return "", fmt.Errorf("missing input key %q", r.queryKey)
	}

	docs, err := r.retriever.GetRelevantDocuments(ctx, fmt.Sprintf("%v", query))
	if err != nil {
		return "", fmt.Errorf("retrieval error: %w", err)
	}

	input["input_documents"] = docs
	return r.chain.Invoke(ctx, input, opts...)
}

// Stream streams the chain output.
func (r *RetrievalQA) Stream(ctx context.Context, input map[string]any, opts ...core.Option) (*core.StreamIterator[string], error) {
	result, err := r.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	ch := make(chan core.StreamChunk[string], 1)
	ch <- core.StreamChunk[string]{Value: result}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

// Batch runs the chain for multiple inputs.
func (r *RetrievalQA) Batch(ctx context.Context, inputs []map[string]any, opts ...core.Option) ([]string, error) {
	results := make([]string, len(inputs))
	for i, input := range inputs {
		result, err := r.Invoke(ctx, input, opts...)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}
