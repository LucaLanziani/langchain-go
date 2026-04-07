package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/provider"
)

// Example 10: Router with Fallback
// Demonstrates using fallback strategies to automatically retry failed requests
// with alternative providers, ensuring high availability.

func main() {
	ctx := context.Background()

	fmt.Println("=== Creating Router with Fallback Strategy ===\n")

	// Create router with multiple providers and sequential fallback
	// Note: Fallback is configured via inline RouterOption
	router, err := provider.NewRouter(
		ctx,
		[]provider.ProviderEntry{
			{
				Name:         "primary-openai",
				ProviderType: provider.ProviderOpenAI,
				Options: []provider.ProviderOption{
					provider.WithModel("gpt-4o-mini"),
					provider.WithTemperature(0.7),
				},
			},
			{
				Name:         "backup-anthropic",
				ProviderType: provider.ProviderAnthropic,
				Options: []provider.ProviderOption{
					provider.WithModel("claude-3-5-haiku-20241022"),
					provider.WithMaxTokens(1000),
				},
			},
			{
				Name:         "local-ollama",
				ProviderType: provider.ProviderOllama,
				Options: []provider.ProviderOption{
					provider.WithModel("llama3.2"),
					provider.WithBaseURL("http://localhost:11434"),
				},
			},
		},
		// Use SimpleStrategy to prefer primary provider
		&provider.SimpleStrategy{ProviderName: "primary-openai"},
		// Configure sequential fallback via inline option
		func(config *provider.RouterConfig) {
			config.FallbackStrategy = &provider.SequentialFallback{
				Order: []string{"primary-openai", "backup-anthropic", "local-ollama"},
			}
		},
	)
	if err != nil {
		log.Fatalf("Failed to create router: %v", err)
	}
	defer router.Cleanup()

	fmt.Println("Router created with fallback chain:")
	fmt.Println("  Primary: primary-openai (gpt-4o-mini)")
	fmt.Println("  Backup 1: backup-anthropic (claude-3-5-haiku)")
	fmt.Println("  Backup 2: local-ollama (llama3.2)")
	fmt.Println("\nIf primary fails → tries backup-anthropic → tries local-ollama")
	fmt.Println()

	// Test 1: Normal request (should succeed with primary)
	fmt.Println("=== Test 1: Normal Request ===")
	fmt.Println("Question: What is the capital of France?")
	response, err := router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("What is the capital of France?"),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Test 2: Another request
	fmt.Println("=== Test 2: Follow-up Request ===")
	fmt.Println("Question: What is its population?")
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("What is the population of Paris?"),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Test 3: Complex request
	fmt.Println("=== Test 3: Complex Request ===")
	fmt.Println("Question: Explain quantum computing")
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("Explain quantum computing in simple terms"),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Display metrics
	fmt.Println("=== Router Metrics ===")
	metrics := router.GetMetrics()
	for name, m := range metrics {
		fmt.Printf("%s:\n", name)
		fmt.Printf("  Requests: %d\n", m.RequestCount)
		fmt.Printf("  Errors: %d\n", m.ErrorCount)
		if m.RequestCount > 0 {
			successRate := float64(m.RequestCount-m.ErrorCount) / float64(m.RequestCount) * 100
			fmt.Printf("  Success Rate: %.1f%%\n", successRate)
			fmt.Printf("  Avg Latency: %v\n", m.TotalLatency/time.Duration(m.RequestCount))
		}
		fmt.Println()
	}

	fmt.Println("=== Example Complete ===")
	fmt.Println("\nFallback behavior:")
	fmt.Println("  - If primary-openai succeeds: request completes immediately")
	fmt.Println("  - If primary-openai fails: automatically tries backup-anthropic")
	fmt.Println("  - If backup-anthropic fails: automatically tries local-ollama")
	fmt.Println("  - If all fail: returns error with details of all attempts")
	fmt.Println("\nThis ensures high availability even when providers have issues.")

	// Example with SmartFallback
	fmt.Println("\n=== Alternative: Smart Fallback ===")
	fmt.Println("SmartFallback uses metrics to choose the best fallback provider")
	fmt.Println("based on success rates and recent performance.")
	fmt.Println("\nCreate router with SmartFallback:")
	fmt.Println(`
  router, err := provider.NewRouter(
      ctx,
      entries,
      strategy,
      func(config *provider.RouterConfig) {
          config.FallbackStrategy = &provider.SmartFallback{}
      },
  )
	`)
	fmt.Println("\nSmartFallback automatically selects the fallback provider with:")
	fmt.Println("  - Highest success rate")
	fmt.Println("  - Recent successful usage")
	fmt.Println("  - Never returns the provider that just failed")
}
