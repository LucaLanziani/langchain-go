package provider

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
)

type streamTestModel struct {
	name       string
	invokeFunc func(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.AIMessage, error)
	streamFunc func(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.StreamIterator[*core.AIMessage], error)
}

func (m *streamTestModel) Invoke(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.AIMessage, error) {
	if m.invokeFunc != nil {
		return m.invokeFunc(ctx, messages, opts...)
	}
	return core.NewAIMessage("ok"), nil
}

func (m *streamTestModel) Stream(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
	if m.streamFunc != nil {
		return m.streamFunc(ctx, messages, opts...)
	}
	ch := make(chan core.StreamChunk[*core.AIMessage], 1)
	ch <- core.StreamChunk[*core.AIMessage]{Value: core.NewAIMessage("ok")}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

func (m *streamTestModel) Batch(ctx context.Context, inputs [][]core.Message, opts ...core.Option) ([]*core.AIMessage, error) {
	results := make([]*core.AIMessage, len(inputs))
	for i, input := range inputs {
		result, err := m.Invoke(ctx, input, opts...)
		if err != nil {
			return nil, err
		}
		results[i] = result
	}
	return results, nil
}

func (m *streamTestModel) Generate(ctx context.Context, messages []core.Message, opts ...core.Option) (*llms.ChatResult, error) {
	msg, err := m.Invoke(ctx, messages, opts...)
	if err != nil {
		return nil, err
	}
	return &llms.ChatResult{Generations: []*llms.ChatGeneration{{Message: msg}}}, nil
}

func (m *streamTestModel) GetName() string                                    { return m.name }
func (m *streamTestModel) BindTools(...llms.ToolDefinition) llms.ChatModel    { return m }
func (m *streamTestModel) WithStructuredOutput(map[string]any) llms.ChatModel { return m }

func TestRouterStreamMetricsRecordCompletion(t *testing.T) {
	model := &streamTestModel{
		name: "stream-mock",
		streamFunc: func(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
			ch := make(chan core.StreamChunk[*core.AIMessage], 2)
			go func() {
				defer close(ch)
				ch <- core.StreamChunk[*core.AIMessage]{Value: core.NewAIMessage("hello")}
				time.Sleep(20 * time.Millisecond)
				ch <- core.StreamChunk[*core.AIMessage]{Value: core.NewAIMessage("world")}
			}()
			return core.NewStreamIterator(ch), nil
		},
	}
	router := &Router{
		providers: map[string]llms.ChatModel{"mock": model},
		strategy:  &mockStrategy{providerName: "mock"},
		metrics:   newRouterMetrics(),
	}

	iter, err := router.Stream(context.Background(), []core.Message{core.NewHumanMessage("hi")})
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}
	chunks, err := iter.Collect()
	if err != nil {
		t.Fatalf("Collect error: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	metrics := waitForMetrics(t, router, "mock", func(m ProviderMetrics) bool { return m.RequestCount == 1 })
	if metrics.ErrorCount != 0 {
		t.Fatalf("expected no stream errors, got %+v", metrics)
	}
	if metrics.CancelledCount != 0 {
		t.Fatalf("expected no cancellations, got %+v", metrics)
	}
	if metrics.TotalLatency < 20*time.Millisecond {
		t.Fatalf("expected end-to-end latency to include stream duration, got %v", metrics.TotalLatency)
	}
}

func TestRouterStreamMetricsRecordMidstreamError(t *testing.T) {
	model := &streamTestModel{
		name: "stream-mock",
		streamFunc: func(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
			ch := make(chan core.StreamChunk[*core.AIMessage], 2)
			go func() {
				defer close(ch)
				ch <- core.StreamChunk[*core.AIMessage]{Value: core.NewAIMessage("hello")}
				ch <- core.StreamChunk[*core.AIMessage]{Err: errors.New("stream failed")}
			}()
			return core.NewStreamIterator(ch), nil
		},
	}
	router := &Router{
		providers: map[string]llms.ChatModel{"mock": model},
		strategy:  &mockStrategy{providerName: "mock"},
		metrics:   newRouterMetrics(),
	}

	iter, err := router.Stream(context.Background(), []core.Message{core.NewHumanMessage("hi")})
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}
	_, err = iter.Collect()
	if err == nil {
		t.Fatal("expected midstream error")
	}

	metrics := waitForMetrics(t, router, "mock", func(m ProviderMetrics) bool { return m.ErrorCount == 1 })
	if metrics.RequestCount != 1 {
		t.Fatalf("expected one completed stream metric update, got %+v", metrics)
	}
}

func TestRouterStreamMetricsRecordCancellation(t *testing.T) {
	model := &streamTestModel{
		name: "stream-mock",
		streamFunc: func(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
			return core.NewStreamIterator(make(chan core.StreamChunk[*core.AIMessage])), nil
		},
	}
	router := &Router{
		providers: map[string]llms.ChatModel{"mock": model},
		strategy:  &mockStrategy{providerName: "mock"},
		metrics:   newRouterMetrics(),
	}

	iter, err := router.Stream(context.Background(), []core.Message{core.NewHumanMessage("hi")})
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}
	iter.Close()

	metrics := waitForMetrics(t, router, "mock", func(m ProviderMetrics) bool { return m.CancelledCount == 1 })
	if metrics.ErrorCount != 0 {
		t.Fatalf("expected cancellation to be tracked separately, got %+v", metrics)
	}
}

func TestRouterBatchReturnsPerItemErrors(t *testing.T) {
	model := &streamTestModel{
		name: "invoke-mock",
		invokeFunc: func(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.AIMessage, error) {
			if strings.Contains(messages[0].GetContent(), "fail") {
				return nil, errors.New("boom")
			}
			return core.NewAIMessage("ok"), nil
		},
	}
	router := &Router{
		providers: map[string]llms.ChatModel{"mock": model},
		strategy:  &mockStrategy{providerName: "mock"},
		metrics:   newRouterMetrics(),
	}

	results, err := router.Batch(context.Background(), [][]core.Message{
		{core.NewHumanMessage("ok")},
		{core.NewHumanMessage("fail")},
		{core.NewHumanMessage("ok again")},
	})
	if err == nil {
		t.Fatal("expected batch error")
	}
	batchErr, ok := err.(*BatchError)
	if !ok {
		t.Fatalf("expected *BatchError, got %T", err)
	}
	if batchErr.FailedItems[1] == nil {
		t.Fatalf("expected failed item index 1, got %+v", batchErr.FailedItems)
	}
	if results[0] == nil || results[2] == nil {
		t.Fatalf("expected successful items to be preserved, got %+v", results)
	}
	if results[1] != nil {
		t.Fatalf("expected failed item result to be nil, got %+v", results[1])
	}
}

func TestRouterMetricsStatsIncludeCancellations(t *testing.T) {
	metrics := newRouterMetrics()
	metrics.RequestCount["mock"] = 3
	metrics.ErrorCount["mock"] = 1
	metrics.CancelledCount["mock"] = 1
	metrics.TotalLatency["mock"] = 30 * time.Millisecond
	metrics.LastUsed["mock"] = time.Now()

	stats := metrics.GetStats("mock")
	if stats == nil {
		t.Fatal("expected stats for provider")
	}
	if stats.CancelledCount != 1 {
		t.Fatalf("expected cancelled count 1, got %+v", stats)
	}
	if stats.SuccessCount != 1 {
		t.Fatalf("expected success count 1, got %+v", stats)
	}
	if stats.CancelledRate <= 0 {
		t.Fatalf("expected cancelled rate to be tracked, got %+v", stats)
	}

	metrics.Reset("mock")
	if stats := metrics.GetStats("mock"); stats != nil {
		t.Fatalf("expected reset provider stats to be cleared, got %+v", stats)
	}

	metrics.RequestCount["other"] = 1
	metrics.CancelledCount["other"] = 1
	metrics.ResetAll()
	if len(metrics.CancelledCount) != 0 {
		t.Fatalf("expected reset all to clear cancellations, got %+v", metrics.CancelledCount)
	}
}

func waitForMetrics(t *testing.T, router *Router, providerName string, ready func(ProviderMetrics) bool) ProviderMetrics {
	t.Helper()
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		metrics := router.GetMetrics()[providerName]
		if ready(metrics) {
			return metrics
		}
		time.Sleep(10 * time.Millisecond)
	}
	return router.GetMetrics()[providerName]
}
