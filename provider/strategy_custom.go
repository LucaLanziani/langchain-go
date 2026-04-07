package provider

import (
	"context"
	"fmt"
	"time"

	"github.com/LucaLanziani/langchain-go/llms"
)

// SelectProvider executes the user-provided selection function.
// Panics are recovered and returned as errors.
func (s *CustomStrategy) SelectProvider(ctx context.Context, reqCtx RequestContext, providers map[string]llms.ChatModel) (providerName string, err error) {
	// Recover from panics in user code
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("custom strategy panicked: %v", r)
			providerName = ""
		}
	}()

	if s.SelectFunc == nil {
		return "", fmt.Errorf("custom strategy SelectFunc is nil")
	}

	return s.SelectFunc(ctx, reqCtx, providers)
}

// OnSuccess calls the user-provided success callback if defined.
// Panics are recovered and logged but don't propagate.
func (s *CustomStrategy) OnSuccess(ctx context.Context, providerName string, latency time.Duration) {
	// Recover from panics in user code
	defer func() {
		if r := recover(); r != nil {
			// Log panic but don't propagate
			// In production, this should use proper logging
			_ = r
		}
	}()

	if s.OnSuccessFunc != nil {
		s.OnSuccessFunc(ctx, providerName, latency)
	}
}

// OnError calls the user-provided error callback if defined.
// Panics are recovered and logged but don't propagate.
func (s *CustomStrategy) OnError(ctx context.Context, providerName string, err error) {
	// Recover from panics in user code
	defer func() {
		if r := recover(); r != nil {
			// Log panic but don't propagate
			// In production, this should use proper logging
			_ = r
		}
	}()

	if s.OnErrorFunc != nil {
		s.OnErrorFunc(ctx, providerName, err)
	}
}
