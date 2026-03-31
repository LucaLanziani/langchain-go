package runnable

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"net"
	"strings"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
)

// RetryRunnable wraps a Runnable and retries transient failures.
type RetryRunnable[I, O any] struct {
	inner  core.Runnable[I, O]
	config retryConfig
}

type retryConfig struct {
	MaxAttempts       int
	InitialBackoff    time.Duration
	MaxBackoff        time.Duration
	BackoffMultiplier float64
	Jitter            bool
	RetryableError    func(error) bool
}

// RetryOption configures retry behavior.
type RetryOption func(*retryConfig)

// WithRetry wraps a runnable with retry middleware.
func WithRetry[I, O any](inner core.Runnable[I, O], opts ...RetryOption) *RetryRunnable[I, O] {
	cfg := defaultRetryConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	return &RetryRunnable[I, O]{
		inner:  inner,
		config: cfg,
	}
}

func defaultRetryConfig() retryConfig {
	return retryConfig{
		MaxAttempts:       3,
		InitialBackoff:    100 * time.Millisecond,
		MaxBackoff:        5 * time.Second,
		BackoffMultiplier: 2.0,
		Jitter:            true,
		RetryableError:    DefaultRetryableError,
	}
}

// RetryMaxAttempts sets the maximum number of attempts (including the first attempt).
func RetryMaxAttempts(n int) RetryOption {
	return func(c *retryConfig) {
		if n > 0 {
			c.MaxAttempts = n
		}
	}
}

// RetryInitialBackoff sets the initial backoff duration.
func RetryInitialBackoff(d time.Duration) RetryOption {
	return func(c *retryConfig) {
		if d > 0 {
			c.InitialBackoff = d
		}
	}
}

// RetryMaxBackoff sets the maximum backoff duration.
func RetryMaxBackoff(d time.Duration) RetryOption {
	return func(c *retryConfig) {
		if d > 0 {
			c.MaxBackoff = d
		}
	}
}

// RetryBackoffMultiplier sets the exponential backoff multiplier.
func RetryBackoffMultiplier(f float64) RetryOption {
	return func(c *retryConfig) {
		if f >= 1.0 {
			c.BackoffMultiplier = f
		}
	}
}

// RetryJitter enables or disables jitter.
func RetryJitter(enabled bool) RetryOption {
	return func(c *retryConfig) {
		c.Jitter = enabled
	}
}

// RetryOn sets a custom retryable-error predicate.
func RetryOn(fn func(error) bool) RetryOption {
	return func(c *retryConfig) {
		if fn != nil {
			c.RetryableError = fn
		}
	}
}

// DefaultRetryableError classifies common transient errors.
func DefaultRetryableError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}

	var netErr net.Error
	if errors.As(err, &netErr) {
		if netErr.Timeout() {
			return true
		}
	}

	msg := strings.ToLower(err.Error())
	retryableSubstrings := []string{
		"status 429", "status code 429", "status code: 429", "http 429", "too many requests",
		"status 500", "status code 500", "status code: 500", "http 500",
		"status 502", "status code 502", "status code: 502", "http 502",
		"status 503", "status code 503", "status code: 503", "http 503",
		"status 504", "status code 504", "status code: 504", "http 504",
		"connection reset", "connection refused", "broken pipe", "unexpected eof", "timeout",
	}
	for _, substring := range retryableSubstrings {
		if strings.Contains(msg, substring) {
			return true
		}
	}

	return false
}

func (r *RetryRunnable[I, O]) GetName() string {
	return fmt.Sprintf("Retry(%s)", r.inner.GetName())
}

func (r *RetryRunnable[I, O]) Invoke(ctx context.Context, input I, opts ...core.Option) (O, error) {
	cfg := core.ApplyOptions(opts...)
	return doWithRetry(ctx, r.config, r.inner.GetName(), cfg.Callbacks, func() (O, error) {
		return r.inner.Invoke(ctx, input, opts...)
	})
}

func (r *RetryRunnable[I, O]) Stream(ctx context.Context, input I, opts ...core.Option) (*core.StreamIterator[O], error) {
	cfg := core.ApplyOptions(opts...)
	return doWithRetry(ctx, r.config, r.inner.GetName(), cfg.Callbacks, func() (*core.StreamIterator[O], error) {
		return r.inner.Stream(ctx, input, opts...)
	})
}

func (r *RetryRunnable[I, O]) Batch(ctx context.Context, inputs []I, opts ...core.Option) ([]O, error) {
	cfg := core.ApplyOptions(opts...)
	results := make([]O, len(inputs))

	for i, input := range inputs {
		result, err := doWithRetry(ctx, r.config, r.inner.GetName(), cfg.Callbacks, func() (O, error) {
			return r.inner.Invoke(ctx, input, opts...)
		})
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", i, err)
		}
		results[i] = result
	}

	return results, nil
}

func doWithRetry[T any](ctx context.Context, cfg retryConfig, runnableName string, callbacks []core.CallbackHandler, fn func() (T, error)) (T, error) {
	var zero T
	var lastErr error

	for attempt := 1; attempt <= cfg.MaxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return zero, err
		}

		result, err := fn()
		if err == nil {
			return result, nil
		}
		lastErr = err

		if attempt >= cfg.MaxAttempts || !cfg.RetryableError(err) {
			break
		}

		backoff := calculateBackoff(cfg, attempt)
		for _, cb := range callbacks {
			cb.OnRetry(ctx, core.RetryData{
				Attempt:         attempt + 1,
				Error:           err,
				BackoffDuration: backoff,
				RunnableName:    runnableName,
			})
		}

		timer := time.NewTimer(backoff)
		select {
		case <-ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			return zero, ctx.Err()
		case <-timer.C:
		}
	}

	if lastErr == nil {
		return zero, errors.New("retry failed with unknown error")
	}

	return zero, lastErr
}

func calculateBackoff(cfg retryConfig, retryNumber int) time.Duration {
	base := float64(cfg.InitialBackoff)
	exponent := math.Pow(cfg.BackoffMultiplier, float64(retryNumber-1))
	wait := time.Duration(base * exponent)
	if wait > cfg.MaxBackoff {
		wait = cfg.MaxBackoff
	}
	if cfg.Jitter {
		factor := 0.5 + rand.Float64()
		wait = time.Duration(float64(wait) * factor)
	}
	if wait < 0 {
		return 0
	}
	return wait
}

var _ core.Runnable[any, any] = (*RetryRunnable[any, any])(nil)
