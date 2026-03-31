package runnable

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
)

// RateLimitedRunnable wraps a runnable and applies token-bucket rate limiting.
type RateLimitedRunnable[I, O any] struct {
	inner   core.Runnable[I, O]
	config  rateLimitConfig
	limiter *tokenBucketLimiter
}

type rateLimitConfig struct {
	RPS   float64
	Burst int
}

// RateLimitOption configures rate limiting.
type RateLimitOption func(*rateLimitConfig)

// WithRateLimit wraps a runnable with rate limiting middleware.
func WithRateLimit[I, O any](inner core.Runnable[I, O], opts ...RateLimitOption) *RateLimitedRunnable[I, O] {
	cfg := defaultRateLimitConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	r := &RateLimitedRunnable[I, O]{
		inner:  inner,
		config: cfg,
	}
	if cfg.RPS > 0 {
		burst := cfg.Burst
		if burst <= 0 {
			burst = 1
		}
		r.limiter = newTokenBucketLimiter(cfg.RPS, burst)
	}

	return r
}

func defaultRateLimitConfig() rateLimitConfig {
	return rateLimitConfig{
		RPS:   0,
		Burst: 1,
	}
}

// RateLimitRPS sets requests-per-second.
func RateLimitRPS(n float64) RateLimitOption {
	return func(c *rateLimitConfig) {
		if n > 0 {
			c.RPS = n
		}
	}
}

// RateLimitBurst sets burst capacity.
func RateLimitBurst(n int) RateLimitOption {
	return func(c *rateLimitConfig) {
		if n > 0 {
			c.Burst = n
		}
	}
}

func (r *RateLimitedRunnable[I, O]) GetName() string {
	return fmt.Sprintf("RateLimit(%s)", r.inner.GetName())
}

func (r *RateLimitedRunnable[I, O]) wait(ctx context.Context) error {
	if r.limiter == nil {
		return nil
	}
	return r.limiter.Wait(ctx)
}

func (r *RateLimitedRunnable[I, O]) Invoke(ctx context.Context, input I, opts ...core.Option) (O, error) {
	if err := r.wait(ctx); err != nil {
		var zero O
		return zero, err
	}
	return r.inner.Invoke(ctx, input, opts...)
}

func (r *RateLimitedRunnable[I, O]) Stream(ctx context.Context, input I, opts ...core.Option) (*core.StreamIterator[O], error) {
	if err := r.wait(ctx); err != nil {
		return nil, err
	}
	return r.inner.Stream(ctx, input, opts...)
}

func (r *RateLimitedRunnable[I, O]) Batch(ctx context.Context, inputs []I, opts ...core.Option) ([]O, error) {
	cfg := core.ApplyOptions(opts...)
	results := make([]O, len(inputs))

	maxConcurrency := len(inputs)
	if cfg.MaxConcurrency > 0 && cfg.MaxConcurrency < maxConcurrency {
		maxConcurrency = cfg.MaxConcurrency
	}
	if maxConcurrency <= 0 {
		maxConcurrency = 1
	}
	sem := make(chan struct{}, maxConcurrency)

	var wg sync.WaitGroup
	var firstErr error
	var errMu sync.Mutex

	for i, input := range inputs {
		wg.Add(1)
		go func(i int, input I) {
			defer wg.Done()

			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				errMu.Lock()
				if firstErr == nil {
					firstErr = ctx.Err()
				}
				errMu.Unlock()
				return
			}
			defer func() { <-sem }()

			if err := r.wait(ctx); err != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				errMu.Unlock()
				return
			}

			result, err := r.inner.Invoke(ctx, input, opts...)
			if err != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("batch item %d: %w", i, err)
				}
				errMu.Unlock()
				return
			}
			results[i] = result
		}(i, input)
	}

	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	return results, nil
}

type tokenBucketLimiter struct {
	rate   float64
	burst  float64
	tokens float64
	last   time.Time
	now    func() time.Time
	mu     sync.Mutex
}

func newTokenBucketLimiter(rate float64, burst int) *tokenBucketLimiter {
	now := time.Now()
	return &tokenBucketLimiter{
		rate:   rate,
		burst:  float64(burst),
		tokens: float64(burst),
		last:   now,
		now:    time.Now,
	}
}

func (l *tokenBucketLimiter) Wait(ctx context.Context) error {
	for {
		d := l.reserveDelay()
		if d <= 0 {
			return nil
		}

		timer := time.NewTimer(d)
		select {
		case <-ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			return ctx.Err()
		case <-timer.C:
		}
	}
}

func (l *tokenBucketLimiter) reserveDelay() time.Duration {
	l.mu.Lock()
	defer l.mu.Unlock()

	now := l.now()
	elapsed := now.Sub(l.last).Seconds()
	if elapsed > 0 {
		l.tokens = math.Min(l.burst, l.tokens+elapsed*l.rate)
		l.last = now
	}

	if l.tokens >= 1 {
		l.tokens--
		return 0
	}

	missing := 1 - l.tokens
	waitSeconds := missing / l.rate
	wait := time.Duration(waitSeconds * float64(time.Second))
	if wait < time.Millisecond {
		wait = time.Millisecond
	}
	return wait
}

var _ core.Runnable[any, any] = (*RateLimitedRunnable[any, any])(nil)
