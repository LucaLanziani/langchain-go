package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/provider"
)

// Example 4: Round-Robin Load Balancing
// Demonstrates using RoundRobinStrategy to distribute requests evenly
// across multiple providers in a circular pattern.

func main() {
	ctx := context.Background()

	fmt.Println("=== Creating Router with Round-Robin Strategy ===\n")

	// Create router with three providers
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
		// RoundRobinStrategy distributes requests evenly
		&provider.RoundRobinStrategy{},
	)
	if err != nil {
		log.Fatalf("Failed to create router: %v", err)
	}
	defer router.Cleanup()

	fmt.Println("Router created with 3 providers:")
	fmt.Println("  - openai-1")
	fmt.Println("  - openai-2")
	fmt.Println("  - anthropic-1")
	fmt.Println("\nUsing RoundRobinStrategy for even distribution\n")

	// Make 9 requests to see round-robin distribution
	// Expected pattern: openai-1, openai-2, anthropic-1, openai-1, openai-2, anthropic-1, ...
	for i := 1; i <= 9; i++ {
		question := fmt.Sprintf("What is %d + %d?", i, i)
		fmt.Printf("Request %d: %s\n", i, question)

		response, err := router.Invoke(ctx, []core.Message{
			core.NewHumanMessage(question),
		})
		if err != nil {
			log.Printf("Request failed: %v\n", err)
			continue
		}

		fmt.Printf("Response: %s\n\n", response.GetContent())
	}

	// Display metrics to show even distribution
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
	fmt.Println("Note: Each provider received approximately 33% of requests")
}
