package core

import (
	"context"
	"fmt"
	"sync"
)

// Batch runs invoke for each input in parallel, honoring MaxConcurrency.
func Batch[I, O any](ctx context.Context, inputs []I, opts []Option, invoke func(context.Context, I, ...Option) (O, error)) ([]O, error) {
	results := make([]O, len(inputs))
	if len(inputs) == 0 {
		return results, nil
	}

	cfg := ApplyOptions(opts...)
	limit := len(inputs)
	if cfg.MaxConcurrency > 0 && cfg.MaxConcurrency < limit {
		limit = cfg.MaxConcurrency
	}
	if limit <= 0 {
		limit = 1
	}

	sem := make(chan struct{}, limit)
	errCh := make(chan error, len(inputs))
	var wg sync.WaitGroup

	for i, input := range inputs {
		wg.Add(1)
		go func(idx int, item I) {
			defer wg.Done()

			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				errCh <- fmt.Errorf("batch item %d: %w", idx, ctx.Err())
				return
			}
			defer func() { <-sem }()

			result, err := invoke(ctx, item, opts...)
			if err != nil {
				errCh <- fmt.Errorf("batch item %d: %w", idx, err)
				return
			}
			results[idx] = result
		}(i, input)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		if err != nil {
			var zero []O
			return zero, err
		}
	}

	return results, nil
}
