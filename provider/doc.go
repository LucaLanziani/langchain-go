/*
Package provider implements a unified interface for creating and managing multiple LLM providers
with intelligent routing, fallback strategies, and metrics tracking.

# Overview

The provider package provides a consistent way to create and configure any of the supported
LLM providers (Anthropic, GitHub Copilot, Ollama, OpenAI) while maintaining backward compatibility
and supporting all provider-specific features. It also includes a Router component for managing
multiple providers with configurable routing and fallback strategies.

# Supported Providers

  - Anthropic (Claude models)
  - GitHub Copilot
  - Ollama (local models)
  - OpenAI (GPT models)

# Single Provider Usage

The simplest way to create a provider is using the factory pattern with NewProvider:

	model, cleanup, err := provider.NewProvider(ctx, provider.ProviderOpenAI,
		provider.WithModel("gpt-4o"),
		provider.WithTemperature(0.7),
		provider.WithAPIKey("sk-..."),
	)
	if err != nil {
		return err
	}
	defer cleanup()

	response, err := model.Invoke(ctx, messages)

The factory returns three values:
  - A ChatModel instance that implements the llms.ChatModel interface
  - A cleanup function that must be called to release resources
  - An error if creation fails

# Common Configuration Options

All providers support these common options:

	WithModel(model string)              // Model name (e.g., "gpt-4o", "claude-3-opus")
	WithTemperature(temp float64)        // Temperature (0.0-2.0)
	WithMaxTokens(tokens int)            // Maximum tokens to generate
	WithTopP(topP float64)               // Top-p sampling (0.0-1.0)
	WithStop(sequences []string)         // Stop sequences
	WithAPIKey(key string)               // API key for authentication
	WithBaseURL(url string)              // Custom base URL

# Provider-Specific Configuration

Provider-specific options are set using WithProviderSpecific:

	// GitHub Copilot with tools
	model, cleanup, err := provider.NewProvider(ctx, provider.ProviderGitHubCopilot,
		provider.WithModel("gpt-4o"),
		provider.WithProviderSpecific("tools", []tools.Tool{myTool}),
		provider.WithProviderSpecific("cli_path", "/usr/local/bin/github-copilot-cli"),
	)

	// Ollama with custom options
	model, cleanup, err := provider.NewProvider(ctx, provider.ProviderOllama,
		provider.WithModel("llama2"),
		provider.WithProviderSpecific("keep_alive", "5m"),
		provider.WithProviderSpecific("num_ctx", 4096),
	)

# Multi-Provider Router

The Router manages multiple provider instances and routes requests based on configurable strategies:

	router, err := provider.NewRouter(ctx,
		[]provider.ProviderEntry{
			{
				Name:         "fast-gpt",
				ProviderType: provider.ProviderOpenAI,
				Options:      []provider.ProviderOption{provider.WithModel("gpt-3.5-turbo")},
				Weight:       3,
				Tags:         []string{"fast", "cheap"},
			},
			{
				Name:         "smart-claude",
				ProviderType: provider.ProviderAnthropic,
				Options:      []provider.ProviderOption{provider.WithModel("claude-3-opus-20240229")},
				Weight:       1,
				Tags:         []string{"smart", "expensive"},
			},
		},
		&provider.WeightedStrategy{},
		provider.WithFallback(&provider.SequentialFallback{Order: []string{"fast-gpt", "smart-claude"}}),
	)
	if err != nil {
		return err
	}
	defer router.Cleanup()

	// Use router like any ChatModel
	response, err := router.Invoke(ctx, messages)

# Routing Strategies

The package provides seven built-in routing strategies:

1. SimpleStrategy - Always routes to a specific provider:

	strategy := &provider.SimpleStrategy{ProviderName: "fast-gpt"}

2. RoundRobinStrategy - Distributes requests evenly across providers:

	strategy := &provider.RoundRobinStrategy{}

3. WeightedStrategy - Routes based on provider weights:

	strategy := provider.NewWeightedStrategy(map[string]int{
		"fast-gpt":     3,  // 75% of requests
		"smart-claude": 1,  // 25% of requests
	})

4. RuleBasedStrategy - Routes based on request characteristics:

	strategy := provider.NewRuleBasedStrategy(
		[]provider.RoutingRule{
			{
				Name:     "complex-to-claude",
				Priority: 100,
				Condition: func(ctx provider.RequestContext) bool {
					return ctx.Complexity == "complex" || ctx.HasToolCalls
				},
				Provider: "smart-claude",
			},
			{
				Name:     "simple-to-gpt",
				Priority: 50,
				Condition: func(ctx provider.RequestContext) bool {
					return ctx.Complexity == "simple"
				},
				Provider: "fast-gpt",
			},
		},
		"fast-gpt", // default provider
	)

5. LoadBalancedStrategy - Routes based on performance metrics:

	strategy := provider.NewLoadBalancedStrategy(router.GetMetrics())

This strategy automatically selects the provider with the best combination of:
  - Lowest average latency
  - Lowest error rate
  - Least recent usage (for load distribution)

6. LLMRoutingStrategy - Uses an LLM to make intelligent routing decisions:

	metaModel, cleanup, _ := provider.NewProvider(ctx, provider.ProviderOpenAI,
		provider.WithModel("gpt-4o-mini"),
	)
	defer cleanup()

	strategy := provider.NewLLMRoutingStrategy(
		metaModel,
		[]string{"fast-gpt", "smart-claude"},
		map[string]string{
			"fast-gpt":     "Fast and cost-effective for simple queries",
			"smart-claude": "Best for complex reasoning and analysis",
		},
	)

The LLM analyzes each request and selects the most appropriate provider. Decisions are cached
to avoid repeated LLM calls for similar requests.

7. CustomStrategy - User-defined routing logic:

	strategy := &provider.CustomStrategy{
		SelectFunc: func(ctx context.Context, reqCtx provider.RequestContext, providers map[string]llms.ChatModel) (string, error) {
			// Custom logic here
			if reqCtx.TotalTokens > 5000 {
				return "smart-claude", nil
			}
			return "fast-gpt", nil
		},
		OnSuccessFunc: func(ctx context.Context, providerName string, latency time.Duration) {
			// Track successes
		},
		OnErrorFunc: func(ctx context.Context, providerName string, err error) {
			// Track errors
		},
	}

# Fallback Strategies

When a provider fails, the router can automatically try alternative providers:

1. NoFallback - Never retries (default):

	provider.WithFallback(&provider.NoFallback{})

2. SequentialFallback - Tries providers in order:

	provider.WithFallback(&provider.SequentialFallback{
		Order: []string{"fast-gpt", "smart-claude", "local-ollama"},
	})

3. SmartFallback - Selects fallback based on success rate:

	provider.WithFallback(&provider.SmartFallback{})

SmartFallback uses metrics to choose the provider with the highest success rate,
giving preference to recently successful providers.

# Metrics Tracking

The router automatically tracks metrics for all providers:

	metrics := router.GetMetrics()
	for name, m := range metrics {
		fmt.Printf("Provider: %s\n", name)
		fmt.Printf("  Requests: %d\n", m.RequestCount)
		fmt.Printf("  Errors: %d\n", m.ErrorCount)
		fmt.Printf("  Avg Latency: %v\n", m.TotalLatency/time.Duration(m.RequestCount))
		fmt.Printf("  Last Used: %v\n", m.LastUsed)
	}

Metrics include:
  - Request count per provider
  - Error count per provider
  - Total latency per provider
  - Last usage timestamp per provider

# Request Context

The router builds a RequestContext for each request, which routing strategies use to make decisions:

	type RequestContext struct {
		Messages     []core.Message  // The messages being sent
		MessageCount int             // Number of messages
		TotalTokens  int             // Estimated token count
		HasToolCalls bool            // Whether request involves tool calls
		Priority     string          // "low", "medium", "high"
		Complexity   string          // "simple", "moderate", "complex"
		UserMetadata map[string]any  // Custom metadata
	}

Complexity is automatically inferred:
  - "complex": >10,000 tokens or has tool calls
  - "simple": <1,000 tokens and no tool calls
  - "moderate": everything else

# Authentication

The package supports multiple authentication methods:

1. Explicit API key:

	provider.WithAPIKey("sk-...")

2. Environment variables:

	ANTHROPIC_API_KEY  // For Anthropic
	OPENAI_API_KEY     // For OpenAI
	GITHUB_TOKEN       // For GitHub Copilot

3. GitHub CLI (for Copilot):

	gh auth token

Ollama doesn't require authentication.

# Resource Cleanup

Always call cleanup functions to release resources:

	// Single provider
	model, cleanup, err := provider.NewProvider(ctx, provider.ProviderOpenAI, ...)
	if err != nil {
		return err
	}
	defer cleanup()

	// Router
	router, err := provider.NewRouter(ctx, entries, strategy)
	if err != nil {
		return err
	}
	defer router.Cleanup()

For most providers, cleanup is a no-op. For GitHub Copilot, cleanup stops the CLI server process.

# Thread Safety

All components are thread-safe and support concurrent usage:
  - Multiple goroutines can invoke the same provider or router concurrently
  - Metrics updates are atomic
  - Routing strategy state is protected by mutexes

# Error Handling

The package provides detailed error types:

	err := provider.NewProviderError(providerType, name, operation, cause)
	err := provider.NewRoutingError(operation, cause)
	err := provider.NewFallbackError(failedProvider, attemptedProviders, cause)

Common errors:
  - ErrUnknownProvider: Invalid provider type
  - ErrProviderNotFound: Provider name not found in router
  - ErrNoProvidersAvailable: Router has no providers
  - ErrNoFallbackAvailable: No fallback provider available
  - ErrRouterClosed: Router has been cleaned up

# Complete Example

	package main

	import (
		"context"
		"fmt"
		"log"

		"github.com/LucaLanziani/langchain-go/core"
		"github.com/LucaLanziani/langchain-go/provider"
	)

	func main() {
		ctx := context.Background()

		// Create router with multiple providers
		router, err := provider.NewRouter(ctx,
			[]provider.ProviderEntry{
				{
					Name:         "fast",
					ProviderType: provider.ProviderOpenAI,
					Options: []provider.ProviderOption{
						provider.WithModel("gpt-3.5-turbo"),
						provider.WithTemperature(0.7),
					},
					Weight: 3,
				},
				{
					Name:         "smart",
					ProviderType: provider.ProviderAnthropic,
					Options: []provider.ProviderOption{
						provider.WithModel("claude-3-opus-20240229"),
						provider.WithMaxTokens(4096),
					},
					Weight: 1,
				},
			},
			provider.NewWeightedStrategy(map[string]int{
				"fast":  3,
				"smart": 1,
			}),
			provider.WithFallback(&provider.SequentialFallback{
				Order: []string{"fast", "smart"},
			}),
		)
		if err != nil {
			log.Fatal(err)
		}
		defer router.Cleanup()

		// Use router like any ChatModel
		messages := []core.Message{
			core.NewHumanMessage("What is the capital of France?"),
		}

		response, err := router.Invoke(ctx, messages)
		if err != nil {
			log.Fatal(err)
		}

		fmt.Println(response.GetContent())

		// Check metrics
		metrics := router.GetMetrics()
		for name, m := range metrics {
			fmt.Printf("%s: %d requests, %d errors\n", name, m.RequestCount, m.ErrorCount)
		}
	}

# Backward Compatibility

The package maintains full backward compatibility with existing provider-specific constructors:

	// Still works
	model := anthropic.New(anthropic.WithModelName("claude-3-opus-20240229"))
	model := openai.New(openai.WithModelName("gpt-4o"))
	model := ollama.New(ollama.WithModel("llama2"))
	model, _ := copilot.New(ctx, copilot.WithModelName("gpt-4o"))

The unified interface is an addition, not a replacement.
*/
package provider
