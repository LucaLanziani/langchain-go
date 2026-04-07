package provider

// This file serves as the main entry point for the provider package.
// It re-exports all public types and functions for convenient access.
//
// For detailed documentation, see doc.go or the package documentation at:
// https://pkg.go.dev/github.com/LucaLanziani/langchain-go/provider

// ===== Core Factory Functions =====
// NewProvider and NewRouter are the primary entry points for creating providers and routers.
// See factory.go and router.go for implementation details.

// ===== Configuration Types =====
// ProviderType, ProviderConfig, ProviderOption, and CleanupFunc are defined in types.go

// ===== Router Types =====
// Router, ProviderEntry, RouterConfig, RouterOption, RequestContext, and RoutingRule
// are defined in types.go

// ===== Strategy Types =====
// RoutingStrategy, FallbackStrategy, and all built-in strategy implementations
// (SimpleStrategy, RoundRobinStrategy, WeightedStrategy, RuleBasedStrategy,
// LoadBalancedStrategy, CustomStrategy, LLMRoutingStrategy) are defined in types.go
// and their respective strategy_*.go files.

// ===== Fallback Types =====
// NoFallback, SequentialFallback, and SmartFallback are defined in types.go
// and their respective fallback_*.go files.

// ===== Metrics Types =====
// RouterMetrics, ProviderMetrics, and ProviderStats are defined in types.go and metrics.go

// ===== Error Types =====
// All error types and sentinel errors are defined in errors.go

// ===== Configuration Options =====
// Common options (WithModel, WithTemperature, etc.) are defined in config.go
// Provider-specific options are defined in options.go

// Version information
const (
	// Version is the current version of the provider package
	Version = "1.0.0"
)
