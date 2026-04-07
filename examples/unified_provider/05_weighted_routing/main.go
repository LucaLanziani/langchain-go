package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/provider"
)

// Example 5: Weighted Routing
// Demonstrates using WeightedStrategy to distribute requests according to
// provider weights. Useful for preferring faster/cheaper providers while
// keeping expensive/powerful providers available.

func main() {
	ctx := context.Background()

	fmt.Println("=== Creating Router with Weighted Strategy ===\n")

	// Create router with three providers and different weights
	router, err := provider.NewRouter(
		ctx,
		[]provider.ProviderEntry{
			{
				Name:         "fast-gpt35",
				ProviderType: provider.ProviderOpenAI,
				Options: []provider.ProviderOption{
					provider.WithModel("gpt-3.5-turbo"),
					provider.WithTemperature(0.7),
				},
				Weight: 70, // 70% of requests
			},
			{
				Name:         "smart-gpt4",
				ProviderType: provider.ProviderOpenAI,
				Options: []provider.ProviderOption{
					provider.WithModel("gpt-4o-mini"),
					provider.WithTemperature(0.7),
				},
				Weight: 20, // 20% of requests
			},
			{
				Name:         "local-ollama",
				ProviderType: provider.ProviderOllama,
				Options: []provider.ProviderOption{
					provider.WithModel("llama3.2"),
					provider.WithBaseURL("http://localhost:11434"),
				},
				Weight: 10, // 10% of requests
			},
		},
		// WeightedStrategy distributes based on weights
		provider.NewWeightedStrategy(map[string]int{
			"fast-gpt35":   70,
			"smart-gpt4":   20,
			"local-ollama": 10,
		}),
	)
	if err != nil {
		log.Fatalf("Failed to create router: %v", err)
	}
	defer router.Cleanup()

	fmt.Println("Router created with weighted distribution:")
	fmt.Println("  - fast-gpt35 (gpt-3.5-turbo): 70% weight")
	fmt.Println("  - smart-gpt4 (gpt-4o-mini): 20% weight")
	fmt.Println("  - local-ollama (llama3.2): 10% weight")
	fmt.Println("\nMaking 20 requests to observe distribution...\n")

	// Make 20 requests to see weighted distribution
	successCount := 0
	for i := 1; i <= 20; i++ {
		question := fmt.Sprintf("Count to %d", i)

		response, err := router.Invoke(ctx, []core.Message{
			core.NewHumanMessage(question),
		})
		if err != nil {
			log.Printf("Request %d failed: %v\n", i, err)
			continue
		}

		successCount++
		if i%5 == 0 {
			fmt.Printf("Completed %d requests...\n", i)
		}

		// Show a sample response
		if i == 1 {
			fmt.Printf("Sample response: %s\n\n", response.GetContent())
		}
	}

	fmt.Printf("\nCompleted %d successful requests\n\n", successCount)

	// Display metrics to show weighted distribution
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
			fmt.Printf("  Avg Latency: %v\n", m.TotalLatency/time.Duration(m.RequestCount))
		}
		fmt.Println()
	}

	fmt.Println("=== Example Complete ===")
	fmt.Println("Note: Distribution should approximate the configured weights:")
	fmt.Println("  fast-gpt35 ~70%, smart-gpt4 ~20%, local-ollama ~10%")
}
