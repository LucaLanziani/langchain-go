# Feature 001: Retry and Rate Limiting Middleware

## User Story

**As a** developer building a production AI application with langchain-go,
**I want** automatic retry with exponential backoff and rate limiting for LLM calls,
**so that** my application gracefully handles transient API failures, 429 (rate limit) responses, and provider throttling without manual error-handling boilerplate in every call site.

### Acceptance Criteria

- I can wrap any `Runnable[I, O]` with retry middleware using a one-liner.
- Retries use exponential backoff with jitter by default.
- I can configure max attempts, initial/max backoff, and which errors are retryable.
- I can apply a token-bucket or sliding-window rate limiter to any `Runnable`.
- Rate limiting and retry compose together: rate limit first, then retry on transient failures.
- The middleware propagates `context.Context` cancellation — if the context is cancelled mid-backoff, it returns immediately.
- Retry attempts are visible through the callback system (`OnRetry` events).
- The middleware works with `Invoke`, `Stream`, and `Batch`.

### Example Usage

```go
model := openai.New()

// Wrap with retry (exponential backoff, max 3 attempts)
resilient := runnable.WithRetry(model,
    runnable.RetryMaxAttempts(3),
    runnable.RetryInitialBackoff(500 * time.Millisecond),
    runnable.RetryBackoffMultiplier(2.0),
    runnable.RetryOn(IsTransientError), // custom predicate
)

// Wrap with rate limiting (10 requests per second)
limited := runnable.WithRateLimit(resilient,
    runnable.RateLimitRPS(10),
)

// Use exactly like the original Runnable
result, err := limited.Invoke(ctx, messages)
```

---

## Implementation Plan

### New Package: `runnable/retry.go`

1. **`RetryRunnable[I, O]`** — wraps a `Runnable[I, O]`:
   - Holds a `RetryConfig` with: `MaxAttempts int`, `InitialBackoff time.Duration`, `MaxBackoff time.Duration`, `BackoffMultiplier float64`, `Jitter bool`, `RetryableError func(error) bool`.
   - `Invoke`: loop up to MaxAttempts, call inner Runnable, on retryable error sleep with backoff + jitter, respect `ctx.Done()`.
   - `Stream`: same retry logic around the initial `Stream()` call (not mid-stream).
   - `Batch`: retry each item independently (failed items are retried, successful ones are kept).
   - Fire `OnRetry` callback with attempt number and error on each retry.

2. **Default retryable error classifier** — retry on:
   - HTTP 429 (Too Many Requests)
   - HTTP 500, 502, 503, 504
   - Network timeouts / connection resets
   - Any error wrapping the above

3. **Functional options** — `RetryOption` type:
   - `RetryMaxAttempts(n int)`
   - `RetryInitialBackoff(d time.Duration)`
   - `RetryMaxBackoff(d time.Duration)`
   - `RetryBackoffMultiplier(f float64)`
   - `RetryJitter(enabled bool)`
   - `RetryOn(fn func(error) bool)`

### New Package: `runnable/ratelimit.go`

1. **`RateLimitedRunnable[I, O]`** — wraps a `Runnable[I, O]`:
   - Uses a token-bucket algorithm (leverage `golang.org/x/time/rate` or implement in-stdlib with `time.Ticker` + buffered channel).
   - `Invoke`: acquire a token before calling inner Runnable.
   - `Stream`: acquire a token before opening the stream.
   - `Batch`: acquire tokens for each item (respecting `MaxConcurrency`).
   - Blocks until a token is available or `ctx` is cancelled.

2. **Functional options** — `RateLimitOption` type:
   - `RateLimitRPS(n float64)` — requests per second
   - `RateLimitBurst(n int)` — burst capacity

### Callback Integration

Add to `core/callbacks.go`:

```go
OnRetry(ctx context.Context, data RetryData)
```

Where `RetryData` carries: `Attempt int`, `Error error`, `BackoffDuration time.Duration`, `RunnableName string`.

### Testing Strategy

- Unit tests with a `failNTimes` mock Runnable that fails N times then succeeds.
- Verify backoff durations grow exponentially (within jitter tolerance).
- Verify context cancellation aborts retries immediately.
- Verify rate limiter enforces timing (use `time.Now()` injection for deterministic tests).
- Integration test: compose retry + rate limit + real Runnable.

### Dependencies

- No new external dependencies required (`time`, `math/rand`, `sync` from stdlib).
- Optional: `golang.org/x/time/rate` for the rate limiter (well-tested, stdlib-adjacent).
