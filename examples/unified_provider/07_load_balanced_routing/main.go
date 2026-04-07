package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/provider"
)

// Example 7: Load-Balanced Routing
// Demonstrates using LoadBalancedStrategy to automatically route requests
// to the provider with the best performance (lowest latency and error rate).

func main() {
	ctx := context.Background()

	fmt.Println("=== Creating Router with Load-Balanced Strategy ===\n")

	// Create router with multiple providers
	router, err := provider.NewRouter(
		ctx,
		[]provider.ProviderEntry{
			{
				Name:         "openai-1",
				ProviderType: provider.ProviderOpenAI,
				Options: []provider.ProviderOption{
					provider.WithModel("gpt-4o-mini"),
					provider.WithTemperature(0.7),
				},
			},
			{
				Name:         "openai-2",
				ProviderType: provider.ProviderOpenAI,
				Options: []provider.ProviderOption{
					provider.WithModel("gpt-4o-mini"),
					provider.WithTemperature(0.7),
				},
			},
			{
				Name:         "anthropic-1",
				ProviderType: provider.ProviderAnthropic,
				Options: []provider.ProviderOption{
					provider.WithModel("claude-3-5-haiku-20241022"),
					provider.WithMaxTokens(1000),
				},
			},
		},
		// LoadBalancedStrategy automatically selects best performing provider
		provider.NewLoadBalancedStrategy(nil), // Will use router's metrics
	)
	if err != nil {
		log.Fatalf("Failed to create router: %v", err)
	}
	defer router.Cleanup()

	fmt.Println("Router created with load-balanced routing:")
	fmt.Println("  - openai-1 (gpt-4o-mini)")
	fmt.Println("  - openai-2 (gpt-4o-mini)")
	fmt.Println("  - anthropic-1 (claude-3-5-haiku)")
	fmt.Println("\nStrategy automatically selects provider with:")
	fmt.Println("  - Lowest average latency")
	fmt.Println("  - Lowest error rate")
	fmt.Println("  - Balanced load distribution")
	fmt.Println()

	// Make multiple requests and watch the load balancer adapt
	fmt.Println("=== Making 15 Requests ===\n")

	for i := 1; i <= 15; i++ {
		question := fmt.Sprintf("What is %d squared?", i)

		start := time.Now()
		response, err := router.Invoke(ctx, []core.Message{
			core.NewHumanMessage(question),
		})
		latency := time.Since(start)

		if err != nil {
			log.Printf("Request %d failed: %v\n", i, err)
			continue
		}

		fmt.Printf("Request %d: %s (latency: %v)\n", i, question, latency)

		// Show sample responses
		if i == 1 || i == 5 || i == 10 || i == 15 {
			fmt.Printf("  Response: %s\n", response.GetContent())
		}

		// Small delay between requests
		time.Sleep(100 * time.Millisecond)
	}

	fmt.Println()

	// Display detailed metrics
	fmt.Println("=== Router Metrics ===")
	metrics := router.GetMetrics()

	totalRequests := int64(0)
	for _, m := range metrics {
		totalRequests += m.RequestCount
	}

	for name, m := range metrics {
		percentage := float64(m.RequestCount) / float64(totalRequests) * 100
		fmt.Printf("%s:\n", name)
		fmt.Printf("  Requests: %d (%.1f%%)\n", m.RequestCount, percentage)
		fmt.Printf("  Errors: %d\n", m.ErrorCount)
		if m.RequestCount > 0 {
			avgLatency := m.TotalLatency / time.Duration(m.RequestCount)
			errorRate := float64(m.ErrorCount) / float64(m.RequestCount) * 100
			fmt.Printf("  Avg Latency: %v\n", avgLatency)
			fmt.Printf("  Error Rate: %.1f%%\n", errorRate)
			fmt.Printf("  Last Used: %v ago\n", time.Since(m.LastUsed).Round(time.Second))
		}
		fmt.Println()
	}

	fmt.Println("=== Example Complete ===")
	fmt.Println("Note: The load balancer automatically distributed requests")
	fmt.Println("based on performance metrics, favoring faster providers with")
	fmt.Println("lower error rates while maintaining balanced load.")
}
