package chains

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
	"github.com/LucaLanziani/langchain-go/prompts"
)

// mockLLM is a test double for llms.ChatModel.
type mockLLM struct {
	response  string
	err       error
	streamErr error
}

func (m *mockLLM) Invoke(_ context.Context, _ []core.Message, _ ...core.Option) (*core.AIMessage, error) {
	if m.err != nil {
		return nil, m.err
	}
	return core.NewAIMessage(m.response), nil
}

func (m *mockLLM) Stream(_ context.Context, _ []core.Message, _ ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
	if m.err != nil {
		return nil, m.err
	}
	ch := make(chan core.StreamChunk[*core.AIMessage], 2)
	if m.streamErr != nil {
		ch <- core.StreamChunk[*core.AIMessage]{Err: m.streamErr}
	} else {
		ch <- core.StreamChunk[*core.AIMessage]{Value: core.NewAIMessage(m.response)}
	}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

func (m *mockLLM) Batch(_ context.Context, inputs [][]core.Message, _ ...core.Option) ([]*core.AIMessage, error) {
	results := make([]*core.AIMessage, len(inputs))
	for i := range inputs {
		results[i] = core.NewAIMessage(m.response)
	}
	return results, nil
}

func (m *mockLLM) Generate(_ context.Context, _ []core.Message, _ ...core.Option) (*llms.ChatResult, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &llms.ChatResult{
		Generations: []*llms.ChatGeneration{{Message: core.NewAIMessage(m.response)}},
	}, nil
}

func (m *mockLLM) GetName() string                                    { return "MockLLM" }
func (m *mockLLM) BindTools(...llms.ToolDefinition) llms.ChatModel    { return m }
func (m *mockLLM) WithStructuredOutput(map[string]any) llms.ChatModel { return m }

// mockRetriever is a test double for retrievers.Retriever.
type mockRetriever struct {
	docs []*core.Document
	err  error
}

func (r *mockRetriever) GetRelevantDocuments(_ context.Context, _ string) ([]*core.Document, error) {
	return r.docs, r.err
}

func (r *mockRetriever) Invoke(_ context.Context, _ string, _ ...core.Option) ([]*core.Document, error) {
	return r.docs, r.err
}

func (r *mockRetriever) Stream(_ context.Context, _ string, _ ...core.Option) (*core.StreamIterator[[]*core.Document], error) {
	ch := make(chan core.StreamChunk[[]*core.Document], 1)
	ch <- core.StreamChunk[[]*core.Document]{Value: r.docs}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

func (r *mockRetriever) Batch(_ context.Context, inputs []string, _ ...core.Option) ([][]*core.Document, error) {
	results := make([][]*core.Document, len(inputs))
	for i := range inputs {
		results[i] = r.docs
	}
	return results, nil
}

func (r *mockRetriever) GetName() string { return "MockRetriever" }

func newTestChain(response string) (*LLMChain, *mockLLM) {
	llm := &mockLLM{response: response}
	prompt := prompts.NewChatPromptTemplate(
		prompts.Human("{input}"),
	)
	return NewLLMChain(llm, prompt), llm
}

// --- LLMChain tests ---

func TestLLMChainInvoke(t *testing.T) {
	chain, _ := newTestChain("hello world")
	result, err := chain.Invoke(context.Background(), map[string]any{"input": "hi"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "hello world" {
		t.Errorf("expected 'hello world', got %q", result)
	}
}

func TestLLMChainInvokeLLMError(t *testing.T) {
	llm := &mockLLM{err: fmt.Errorf("model unavailable")}
	prompt := prompts.NewChatPromptTemplate(prompts.Human("{input}"))
	chain := NewLLMChain(llm, prompt)
	_, err := chain.Invoke(context.Background(), map[string]any{"input": "hi"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestLLMChainStream(t *testing.T) {
	chain, _ := newTestChain("streamed")
	iter, err := chain.Stream(context.Background(), map[string]any{"input": "hi"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var chunks []string
	for {
		chunk, ok, err := iter.Next()
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		if !ok {
			break
		}
		chunks = append(chunks, chunk)
	}
	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}
}

func TestLLMChainBatch(t *testing.T) {
	chain, _ := newTestChain("ok")
	inputs := []map[string]any{{"input": "a"}, {"input": "b"}}
	results, err := chain.Batch(context.Background(), inputs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
}

func TestLLMChainCallbacks(t *testing.T) {
	var started, ended bool
	cb := &testCallback{
		onStart: func() { started = true },
		onEnd:   func() { ended = true },
	}
	chain, _ := newTestChain("ok")
	_, err := chain.Invoke(context.Background(), map[string]any{"input": "hi"}, core.WithCallbacks(cb))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !started {
		t.Error("OnChainStart was not called")
	}
	if !ended {
		t.Error("OnChainEnd was not called")
	}
}

func TestLLMChainCallbacksOnError(t *testing.T) {
	var errored bool
	cb := &testCallback{onError: func() { errored = true }}
	llm := &mockLLM{err: fmt.Errorf("boom")}
	prompt := prompts.NewChatPromptTemplate(prompts.Human("{input}"))
	chain := NewLLMChain(llm, prompt)
	_, _ = chain.Invoke(context.Background(), map[string]any{"input": "hi"}, core.WithCallbacks(cb))
	if !errored {
		t.Error("OnChainError was not called")
	}
}

func TestLLMChainGetName(t *testing.T) {
	chain, _ := newTestChain("ok")
	if chain.GetName() != "LLMChain" {
		t.Errorf("expected 'LLMChain', got %q", chain.GetName())
	}
}

// --- StuffDocumentsChain tests ---

func TestStuffDocumentsChainInvoke(t *testing.T) {
	chain, _ := newTestChain("answer")
	stuff := NewStuffDocumentsChain(chain)

	docs := []*core.Document{
		{PageContent: "fact one"},
		{PageContent: "fact two"},
	}
	result, err := stuff.Invoke(context.Background(), map[string]any{
		"input_documents": docs,
		"input":           "question",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "answer" {
		t.Errorf("expected 'answer', got %q", result)
	}
}

func TestStuffDocumentsChainMissingKey(t *testing.T) {
	chain, _ := newTestChain("ok")
	stuff := NewStuffDocumentsChain(chain)
	_, err := stuff.Invoke(context.Background(), map[string]any{"input": "q"})
	if err == nil {
		t.Fatal("expected error for missing input_documents key")
	}
}

func TestStuffDocumentsChainWrongType(t *testing.T) {
	chain, _ := newTestChain("ok")
	stuff := NewStuffDocumentsChain(chain)
	_, err := stuff.Invoke(context.Background(), map[string]any{
		"input_documents": "not a slice",
		"input":           "q",
	})
	if err == nil {
		t.Fatal("expected error for wrong type")
	}
}

func TestStuffDocumentsChainStream(t *testing.T) {
	chain, _ := newTestChain("streamed answer")
	stuff := NewStuffDocumentsChain(chain)

	docs := []*core.Document{{PageContent: "context"}}
	iter, err := stuff.Stream(context.Background(), map[string]any{
		"input_documents": docs,
		"input":           "question",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var got strings.Builder
	for {
		chunk, ok, err := iter.Next()
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		if !ok {
			break
		}
		got.WriteString(chunk)
	}
	if got.Len() == 0 {
		t.Error("expected non-empty streamed output")
	}
}

// --- RetrievalQA tests ---

func TestRetrievalQAInvoke(t *testing.T) {
	chain, _ := newTestChain("42")
	retriever := &mockRetriever{
		docs: []*core.Document{{PageContent: "the answer is 42"}},
	}
	qa := NewRetrievalQA(retriever, chain)
	result, err := qa.Invoke(context.Background(), map[string]any{"query": "what is the answer?"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "42" {
		t.Errorf("expected '42', got %q", result)
	}
}

func TestRetrievalQAMissingQueryKey(t *testing.T) {
	chain, _ := newTestChain("ok")
	qa := NewRetrievalQA(&mockRetriever{}, chain)
	_, err := qa.Invoke(context.Background(), map[string]any{})
	if err == nil {
		t.Fatal("expected error for missing query key")
	}
}

func TestRetrievalQARetrieverError(t *testing.T) {
	chain, _ := newTestChain("ok")
	retriever := &mockRetriever{err: fmt.Errorf("retrieval failed")}
	qa := NewRetrievalQA(retriever, chain)
	_, err := qa.Invoke(context.Background(), map[string]any{"query": "something"})
	if err == nil {
		t.Fatal("expected error from retriever")
	}
}

func TestRetrievalQAStream(t *testing.T) {
	chain, _ := newTestChain("streamed result")
	retriever := &mockRetriever{
		docs: []*core.Document{{PageContent: "context"}},
	}
	qa := NewRetrievalQA(retriever, chain)
	iter, err := qa.Stream(context.Background(), map[string]any{"query": "question"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var got strings.Builder
	for {
		chunk, ok, err := iter.Next()
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		if !ok {
			break
		}
		got.WriteString(chunk)
	}
	if got.Len() == 0 {
		t.Error("expected non-empty streamed output")
	}
}

// --- testCallback helper ---

type testCallback struct {
	core.BaseCallbackHandler
	onStart func()
	onEnd   func()
	onError func()
}

func (c *testCallback) OnChainStart(_ context.Context, _ map[string]any, _ string, _ string, _ map[string]any) {
	if c.onStart != nil {
		c.onStart()
	}
}

func (c *testCallback) OnChainEnd(_ context.Context, _ map[string]any, _ string) {
	if c.onEnd != nil {
		c.onEnd()
	}
}

func (c *testCallback) OnChainError(_ context.Context, _ error, _ string) {
	if c.onError != nil {
		c.onError()
	}
}

// --- Additional LLMChain tests ---

func TestLLMChainGetNameCustom(t *testing.T) {
	chain, _ := newTestChain("ok")
	chain.name = "Custom"
	if chain.GetName() != "Custom" {
		t.Errorf("expected 'Custom', got %q", chain.GetName())
	}
}

func TestLLMChainInvokePromptError(t *testing.T) {
	llm := &mockLLM{response: "ok"}
	prompt := prompts.NewChatPromptTemplate(prompts.Placeholder("history"))
	chain := NewLLMChain(llm, prompt)
	_, err := chain.Invoke(context.Background(), map[string]any{"history": 42})
	if err == nil {
		t.Error("expected error from prompt format failure")
	}
}

func TestLLMChainInvokePromptErrorWithCallback(t *testing.T) {
	var errored bool
	cb := &testCallback{onError: func() { errored = true }}
	llm := &mockLLM{response: "ok"}
	prompt := prompts.NewChatPromptTemplate(prompts.Placeholder("history"))
	chain := NewLLMChain(llm, prompt)
	_, _ = chain.Invoke(context.Background(), map[string]any{"history": 42}, core.WithCallbacks(cb))
	if !errored {
		t.Error("expected OnChainError to be called")
	}
}

func TestLLMChainStreamPromptError(t *testing.T) {
	llm := &mockLLM{response: "ok"}
	prompt := prompts.NewChatPromptTemplate(prompts.Placeholder("history"))
	chain := NewLLMChain(llm, prompt)
	_, err := chain.Stream(context.Background(), map[string]any{"history": 42})
	if err == nil {
		t.Error("expected error from prompt format failure in stream")
	}
}

func TestLLMChainStreamLLMError(t *testing.T) {
	llm := &mockLLM{err: fmt.Errorf("stream error")}
	prompt := prompts.NewChatPromptTemplate(prompts.Human("{input}"))
	chain := NewLLMChain(llm, prompt)
	_, err := chain.Stream(context.Background(), map[string]any{"input": "hi"})
	if err == nil {
		t.Error("expected error from LLM stream failure")
	}
}

func TestLLMChainStreamChunkError(t *testing.T) {
	llm := &mockLLM{streamErr: fmt.Errorf("chunk error")}
	prompt := prompts.NewChatPromptTemplate(prompts.Human("{input}"))
	chain := NewLLMChain(llm, prompt)
	iter, err := chain.Stream(context.Background(), map[string]any{"input": "hi"})
	if err != nil {
		t.Fatalf("unexpected error creating stream: %v", err)
	}
	_, _, err = iter.Next()
	if err == nil {
		t.Error("expected error from chunk error in stream")
	}
}

func TestLLMChainBatchError(t *testing.T) {
	llm := &mockLLM{response: "ok"}
	prompt := prompts.NewChatPromptTemplate(prompts.Placeholder("history"))
	chain := NewLLMChain(llm, prompt)
	_, err := chain.Batch(context.Background(), []map[string]any{{"history": 42}})
	if err == nil {
		t.Error("expected error in batch")
	}
}

// --- Additional StuffDocumentsChain tests ---

func TestStuffDocumentsChainGetName(t *testing.T) {
	chain, _ := newTestChain("ok")
	stuff := NewStuffDocumentsChain(chain)
	if stuff.GetName() != "StuffDocumentsChain" {
		t.Errorf("expected 'StuffDocumentsChain', got %q", stuff.GetName())
	}
	stuff.name = "Custom"
	if stuff.GetName() != "Custom" {
		t.Errorf("expected 'Custom', got %q", stuff.GetName())
	}
}

func TestStuffDocumentsChainInvokeWithCallbacks(t *testing.T) {
	var started, ended bool
	cb := &testCallback{
		onStart: func() { started = true },
		onEnd:   func() { ended = true },
	}
	chain, _ := newTestChain("ok")
	stuff := NewStuffDocumentsChain(chain)
	docs := []*core.Document{{PageContent: "context"}}
	_, err := stuff.Invoke(context.Background(), map[string]any{
		"input_documents": docs,
		"input":           "q",
	}, core.WithCallbacks(cb))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !started {
		t.Error("expected OnChainStart to be called")
	}
	if !ended {
		t.Error("expected OnChainEnd to be called")
	}
}

func TestStuffDocumentsChainInvokeMissingKeyWithCallback(t *testing.T) {
	var errored bool
	cb := &testCallback{onError: func() { errored = true }}
	chain, _ := newTestChain("ok")
	stuff := NewStuffDocumentsChain(chain)
	_, _ = stuff.Invoke(context.Background(), map[string]any{"input": "q"}, core.WithCallbacks(cb))
	if !errored {
		t.Error("expected OnChainError to be called")
	}
}

func TestStuffDocumentsChainInvokeLLMError(t *testing.T) {
	llm := &mockLLM{err: fmt.Errorf("llm error")}
	prompt := prompts.NewChatPromptTemplate(prompts.Human("{input}"))
	stuff := NewStuffDocumentsChain(NewLLMChain(llm, prompt))
	docs := []*core.Document{{PageContent: "context"}}
	_, err := stuff.Invoke(context.Background(), map[string]any{
		"input_documents": docs,
		"input":           "q",
	})
	if err == nil {
		t.Error("expected error from LLM failure")
	}
}

func TestStuffDocumentsChainInvokeLLMErrorWithCallback(t *testing.T) {
	var errored bool
	cb := &testCallback{onError: func() { errored = true }}
	llm := &mockLLM{err: fmt.Errorf("llm error")}
	prompt := prompts.NewChatPromptTemplate(prompts.Human("{input}"))
	stuff := NewStuffDocumentsChain(NewLLMChain(llm, prompt))
	docs := []*core.Document{{PageContent: "context"}}
	_, _ = stuff.Invoke(context.Background(), map[string]any{
		"input_documents": docs,
		"input":           "q",
	}, core.WithCallbacks(cb))
	if !errored {
		t.Error("expected OnChainError to be called")
	}
}

func TestStuffDocumentsChainStreamMissingKey(t *testing.T) {
	chain, _ := newTestChain("ok")
	stuff := NewStuffDocumentsChain(chain)
	_, err := stuff.Stream(context.Background(), map[string]any{})
	if err == nil {
		t.Error("expected error when input_documents key is missing")
	}
}

func TestStuffDocumentsChainBatch(t *testing.T) {
	chain, _ := newTestChain("ok")
	stuff := NewStuffDocumentsChain(chain)
	docs := []*core.Document{{PageContent: "ctx"}}
	results, err := stuff.Batch(context.Background(), []map[string]any{
		{"input_documents": docs, "input": "a"},
		{"input_documents": docs, "input": "b"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
}

func TestStuffDocumentsChainBatchError(t *testing.T) {
	chain, _ := newTestChain("ok")
	stuff := NewStuffDocumentsChain(chain)
	_, err := stuff.Batch(context.Background(), []map[string]any{{"input": "no-docs"}})
	if err == nil {
		t.Error("expected error in batch")
	}
}

// --- Additional RetrievalQA tests ---

func TestRetrievalQAGetName(t *testing.T) {
	chain, _ := newTestChain("ok")
	qa := NewRetrievalQA(&mockRetriever{}, chain)
	if qa.GetName() != "RetrievalQA" {
		t.Errorf("expected 'RetrievalQA', got %q", qa.GetName())
	}
	qa.name = "Custom"
	if qa.GetName() != "Custom" {
		t.Errorf("expected 'Custom', got %q", qa.GetName())
	}
}

func TestRetrievalQAInvokeWithCallbacks(t *testing.T) {
	var started, ended bool
	cb := &testCallback{
		onStart: func() { started = true },
		onEnd:   func() { ended = true },
	}
	chain, _ := newTestChain("result")
	retriever := &mockRetriever{docs: []*core.Document{{PageContent: "context"}}}
	qa := NewRetrievalQA(retriever, chain)
	_, err := qa.Invoke(context.Background(), map[string]any{"query": "q"}, core.WithCallbacks(cb))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !started {
		t.Error("expected OnChainStart to be called")
	}
	if !ended {
		t.Error("expected OnChainEnd to be called")
	}
}

func TestRetrievalQAInvokeRetrieverErrorWithCallback(t *testing.T) {
	var errored bool
	cb := &testCallback{onError: func() { errored = true }}
	chain, _ := newTestChain("ok")
	retriever := &mockRetriever{err: fmt.Errorf("retrieval failed")}
	qa := NewRetrievalQA(retriever, chain)
	_, _ = qa.Invoke(context.Background(), map[string]any{"query": "q"}, core.WithCallbacks(cb))
	if !errored {
		t.Error("expected OnChainError to be called")
	}
}

func TestRetrievalQAInvokeChainError(t *testing.T) {
	var errored bool
	cb := &testCallback{onError: func() { errored = true }}
	llm := &mockLLM{err: fmt.Errorf("chain error")}
	prompt := prompts.NewChatPromptTemplate(prompts.Human("{input}"))
	retriever := &mockRetriever{docs: []*core.Document{{PageContent: "ctx"}}}
	qa := NewRetrievalQA(retriever, NewLLMChain(llm, prompt))
	_, _ = qa.Invoke(context.Background(), map[string]any{"query": "q"}, core.WithCallbacks(cb))
	if !errored {
		t.Error("expected OnChainError to be called")
	}
}

func TestRetrievalQAStreamMissingQuery(t *testing.T) {
	chain, _ := newTestChain("ok")
	qa := NewRetrievalQA(&mockRetriever{}, chain)
	_, err := qa.Stream(context.Background(), map[string]any{})
	if err == nil {
		t.Error("expected error when query key is missing")
	}
}

func TestRetrievalQABatch(t *testing.T) {
	chain, _ := newTestChain("ok")
	retriever := &mockRetriever{docs: []*core.Document{{PageContent: "ctx"}}}
	qa := NewRetrievalQA(retriever, chain)
	results, err := qa.Batch(context.Background(), []map[string]any{
		{"query": "q1"},
		{"query": "q2"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
}

func TestRetrievalQABatchError(t *testing.T) {
	chain, _ := newTestChain("ok")
	qa := NewRetrievalQA(&mockRetriever{}, chain)
	_, err := qa.Batch(context.Background(), []map[string]any{{}})
	if err == nil {
		t.Error("expected error in batch")
	}
}
