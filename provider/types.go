package provider

import (
	"context"
	"math/rand"
	"sync"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
)

// ProviderType identifies which LLM provider to use
type ProviderType string

const (
	ProviderAnthropic     ProviderType = "anthropic"
	ProviderGitHubCopilot ProviderType = "github-copilot"
	ProviderOllama        ProviderType = "ollama"
	ProviderOpenAI        ProviderType = "openai"
)

// ProviderConfig holds unified configuration for all providers
type ProviderConfig struct {
	// Common fields (applicable to all providers)
	Model       string
	Temperature *float64
	MaxTokens   *int
	TopP        *float64
	Stop        []string

	// Authentication
	APIKey string // Anthropic, OpenAI, GitHub Copilot (Ollama doesn't need auth)

	// Base URLs
	BaseURL string // Anthropic, OpenAI, Ollama

	// Provider-specific fields
	ProviderSpecific map[string]any
}

// ProviderOption configures the unified provider
type ProviderOption func(*ProviderConfig)

// CleanupFunc releases provider resources (e.g., Copilot CLI server)
type CleanupFunc func() error

// ProviderEntry defines a provider configuration for the router
type ProviderEntry struct {
	Name         string           // Unique identifier (e.g., "fast-openai", "smart-anthropic")
	ProviderType ProviderType     // Type of provider
	Options      []ProviderOption // Configuration options
	Weight       int              // Weight for weighted routing (default: 1)
	Tags         []string         // Tags for categorization (e.g., "fast", "cheap", "smart")
}

// Router manages multiple providers and routes requests between them
// Implements llms.ChatModel interface for transparent usage
type Router struct {
	providers map[string]llms.ChatModel
	cleanups  map[string]CleanupFunc // Map provider name to cleanup function
	strategy  RoutingStrategy
	fallback  FallbackStrategy
	metrics   *RouterMetrics
	mu        sync.RWMutex // Protects providers, cleanups, and router state
}

// RouterConfig holds router configuration
type RouterConfig struct {
	FallbackStrategy FallbackStrategy
	EnableMetrics    bool
	MaxRetries       int
	RetryDelay       time.Duration
}

// RouterOption configures the router
type RouterOption func(*RouterConfig)

// RoutingStrategy determines which provider to use for a request
type RoutingStrategy interface {
	// SelectProvider chooses a provider for the given request
	SelectProvider(ctx context.Context, reqCtx RequestContext, providers map[string]llms.ChatModel) (string, error)

	// OnSuccess is called after a successful request
	OnSuccess(ctx context.Context, providerName string, latency time.Duration)

	// OnError is called after a failed request
	OnError(ctx context.Context, providerName string, err error)
}

// FallbackStrategy determines fallback behavior when primary provider fails
type FallbackStrategy interface {
	// GetFallbackProvider returns next provider to try after failure
	GetFallbackProvider(ctx context.Context, failedProvider string, providers map[string]llms.ChatModel) (string, error)

	// ShouldRetry determines if request should be retried
	ShouldRetry(err error, attemptCount int) bool
}

// RequestContext provides metadata about a request for routing decisions
type RequestContext struct {
	Messages     []core.Message
	MessageCount int
	TotalTokens  int // Estimated
	HasToolCalls bool
	Priority     string // "low", "medium", "high"
	Complexity   string // "simple", "moderate", "complex"
	UserMetadata map[string]any
}

// RoutingRule defines a condition and target provider
type RoutingRule struct {
	Name      string
	Condition func(RequestContext) bool
	Provider  string
	Priority  int // Higher priority rules evaluated first
}

// RouterMetrics tracks routing statistics
type RouterMetrics struct {
	RequestCount map[string]int64         // Requests per provider
	ErrorCount   map[string]int64         // Errors per provider
	TotalLatency map[string]time.Duration // Total latency per provider
	LastUsed     map[string]time.Time     // Last usage timestamp
	mu           sync.RWMutex
}

// SimpleStrategy always routes to a specific provider
type SimpleStrategy struct {
	ProviderName string
}

// RoundRobinStrategy distributes requests evenly across providers
type RoundRobinStrategy struct {
	counter uint64
	mu      sync.Mutex
}

// WeightedStrategy routes based on provider weights
type WeightedStrategy struct {
	weights map[string]int
	mu      sync.RWMutex
	rng     *rand.Rand
}

// RuleBasedStrategy routes based on request characteristics
type RuleBasedStrategy struct {
	rules           []RoutingRule
	defaultProvider string
}

// LoadBalancedStrategy routes based on current load and latency
type LoadBalancedStrategy struct {
	metrics *RouterMetrics
}

// CustomStrategy allows user-defined routing logic
type CustomStrategy struct {
	SelectFunc    func(ctx context.Context, reqCtx RequestContext, providers map[string]llms.ChatModel) (string, error)
	OnSuccessFunc func(ctx context.Context, providerName string, latency time.Duration)
	OnErrorFunc   func(ctx context.Context, providerName string, err error)
}

// LLMRoutingStrategy uses an LLM to make routing decisions
type LLMRoutingStrategy struct {
	model                llms.ChatModel
	providers            []string                  // Available provider names
	systemPrompt         string                    // Custom system prompt for routing
	providerDescriptions map[string]string         // Description of each provider's strengths
	cache                map[string]*llmCacheEntry // Cache routing decisions for similar requests
	cacheTTL             time.Duration             // Cache entry time-to-live
	mu                   sync.RWMutex              // Protects cache
}

// llmCacheEntry stores a routing decision with its expiration time
type llmCacheEntry struct {
	providerName string
	expiresAt    time.Time
}

// NoFallback never retries or falls back
type NoFallback struct{}

// SequentialFallback tries providers in order
type SequentialFallback struct {
	Order []string
}

// SmartFallback uses metrics to choose best fallback
type SmartFallback struct {
	metrics *RouterMetrics
}
