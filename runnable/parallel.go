package runnable

import (
	"context"
	"fmt"
	"sync"

	"github.com/langchain-go/langchain-go/core"
)

// Parallel runs multiple runnables in parallel with the same input,
// collecting their outputs into a map[string]any.
// It implements Runnable[I, map[string]any].
type Parallel[I any] struct {
	branches map[string]func(ctx context.Context, input I, opts ...core.Option) (any, error)
	keys     []string // preserve insertion order
	name     string
}

// NewParallel creates a Parallel runnable from a map of named runnables.
func NewParallel[I, O any](branches map[string]core.Runnable[I, O]) *Parallel[I] {
	p := &Parallel[I]{
		branches: make(map[string]func(ctx context.Context, input I, opts ...core.Option) (any, error)),
	}
	for k, r := range branches {
		r := r // capture
		p.keys = append(p.keys, k)
		p.branches[k] = func(ctx context.Context, input I, opts ...core.Option) (any, error) {
			return r.Invoke(ctx, input, opts...)
		}
	}
	return p
}

// NewParallelAny creates a Parallel runnable from a map of heterogeneous runnables.
func NewParallelAny[I any](branches map[string]func(ctx context.Context, input I, opts ...core.Option) (any, error)) *Parallel[I] {
	p := &Parallel[I]{
		branches: branches,
	}
	for k := range branches {
		p.keys = append(p.keys, k)
	}
	return p
}

// WithName sets the name for tracing.
func (p *Parallel[I]) WithName(name string) *Parallel[I] {
	p.name = name
	return p
}

// GetName returns the name of this parallel runnable.
func (p *Parallel[I]) GetName() string {
	if p.name != "" {
		return p.name
	}
	return "RunnableParallel"
}

// Invoke runs all branches in parallel and collects results into a map.
func (p *Parallel[I]) Invoke(ctx context.Context, input I, opts ...core.Option) (map[string]any, error) {
	cfg := core.ApplyOptions(opts...)

	results := make(map[string]any)
	errs := make(map[string]error)
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Determine concurrency limit.
	sem := make(chan struct{}, len(p.branches))
	if cfg.MaxConcurrency > 0 {
		sem = make(chan struct{}, cfg.MaxConcurrency)
	}

	for _, key := range p.keys {
		key := key
		fn := p.branches[key]
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			result, err := fn(ctx, input, opts...)
			mu.Lock()
			if err != nil {
				errs[key] = err
			} else {
				results[key] = result
			}
			mu.Unlock()
		}()
	}
	wg.Wait()

	if len(errs) > 0 {
		// Return the first error encountered.
		for k, err := range errs {
			return nil, fmt.Errorf("parallel branch %q: %w", k, err)
		}
	}
	return results, nil
}

// Stream invokes and returns the result as a single-chunk stream.
func (p *Parallel[I]) Stream(ctx context.Context, input I, opts ...core.Option) (*core.StreamIterator[map[string]any], error) {
	result, err := p.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	ch := make(chan core.StreamChunk[map[string]any], 1)
	ch <- core.StreamChunk[map[string]any]{Value: result}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

// Batch runs the parallel execution for multiple inputs.
func (p *Parallel[I]) Batch(ctx context.Context, inputs []I, opts ...core.Option) ([]map[string]any, error) {
	results := make([]map[string]any, len(inputs))
	for i, input := range inputs {
		result, err := p.Invoke(ctx, input, opts...)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}
