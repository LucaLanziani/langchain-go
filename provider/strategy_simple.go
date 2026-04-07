package provider

import (
	"context"
	"fmt"
	"time"

	"github.com/LucaLanziani/langchain-go/llms"
)

// SelectProvider returns the configured provider name.
func (s *SimpleStrategy) SelectProvider(ctx context.Context, reqCtx RequestContext, providers map[string]llms.ChatModel) (string, error) {
	if _, exists := providers[s.ProviderName]; !exists {
		return "", fmt.Errorf("%w: %s", ErrProviderNotFound, s.ProviderName)
	}
	return s.ProviderName, nil
}

// OnSuccess is a no-op for SimpleStrategy.
func (s *SimpleStrategy) OnSuccess(ctx context.Context, providerName string, latency time.Duration) {
	// No-op: SimpleStrategy doesn't adapt based on feedback
}

// OnError is a no-op for SimpleStrategy.
func (s *SimpleStrategy) OnError(ctx context.Context, providerName string, err error) {
	// No-op: SimpleStrategy doesn't adapt based on feedback
}
