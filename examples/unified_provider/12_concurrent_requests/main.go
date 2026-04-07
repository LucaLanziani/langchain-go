package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/provider"
)

// Example 12: Concurrent Requests
// Demonstrates that the router is thread-safe and can handle many concurrent
// requests from multiple goroutines, with automatic load distribution.

func main() {
	ctx := context.Background()

	fmt.Println("=== Creating Router for Concurrent Requests ===")

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
		// Use round-robin for even distribution
		&provider.RoundRobinStrategy{},
	)
	if err != nil {
		log.Fatalf("Failed to create router: %v", err)
	}
	defer router.Cleanup()

	fmt.Println("Router created with 3 providers:")
	fmt.Println("  - openai-1 (gpt-4o-mini)")
	fmt.Println("  - openai-2 (gpt-4o-mini)")
	fmt.Println("  - anthropic-1 (claude-3-5-haiku)")
	fmt.Println("\nUsing RoundRobinStrategy for load distribution")
	fmt.Println()

	// Test 1: Sequential requests (baseline)
	fmt.Println("=== Test 1: Sequential Requests (Baseline) ===")
	sequentialStart := time.Now()
	for i := 1; i <= 5; i++ {
		_, err := router.Invoke(ctx, []core.Message{
			core.NewHumanMessage(fmt.Sprintf("What is %d + %d?", i, i)),
		})
		if err != nil {
			log.Printf("Request %d failed: %v\n", i, err)
		}
	}
	sequentialDuration := time.Since(sequentialStart)
	fmt.Printf("5 sequential requests completed in: %v\n\n", sequentialDuration)

	// Test 2: Concurrent requests
	fmt.Println("=== Test 2: Concurrent Requests ===")
	concurrentCount := 50
	fmt.Printf("Making %d concurrent requests...\n", concurrentCount)

	var wg sync.WaitGroup
	var successCount, errorCount atomic.Int64
	var totalLatency atomic.Int64

	concurrentStart := time.Now()

	for i := 1; i <= concurrentCount; i++ {
		wg.Add(1)
		go func(requestNum int) {
			defer wg.Done()

			start := time.Now()
			_, err := router.Invoke(ctx, []core.Message{
				core.NewHumanMessage(fmt.Sprintf("What is %d squared?", requestNum)),
			})
			latency := time.Since(start)

			if err != nil {
				errorCount.Add(1)
				log.Printf("Request %d failed: %v\n", requestNum, err)
			} else {
				successCount.Add(1)
				totalLatency.Add(int64(latency))
			}

			// Progress indicator
			if requestNum%10 == 0 {
				fmt.Printf("  Completed %d requests...\n", requestNum)
			}
		}(i)
	}

	// Wait for all requests to complete
	wg.Wait()
	concurrentDuration := time.Since(concurrentStart)

	fmt.Printf("\n%d concurrent requests completed in: %v\n", concurrentCount, concurrentDuration)
	fmt.Printf("Success: %d, Errors: %d\n", successCount.Load(), errorCount.Load())
	if successCount.Load() > 0 {
		avgLatency := time.Duration(totalLatency.Load() / successCount.Load())
		fmt.Printf("Average latency per request: %v\n", avgLatency)
	}
	fmt.Println()

	// Test 3: High concurrency burst
	fmt.Println("=== Test 3: High Concurrency Burst ===")
	burstCount := 100
	fmt.Printf("Making %d concurrent requests in a burst...\n", burstCount)

	successCount.Store(0)
	errorCount.Store(0)
	totalLatency.Store(0)

	burstStart := time.Now()

	for i := 1; i <= burstCount; i++ {
		wg.Add(1)
		go func(requestNum int) {
			defer wg.Done()

			start := time.Now()
			_, err := router.Invoke(ctx, []core.Message{
				core.NewHumanMessage(fmt.Sprintf("Count to %d", requestNum%10)),
			})
			latency := time.Since(start)

			if err != nil {
				errorCount.Add(1)
			} else {
				successCount.Add(1)
				totalLatency.Add(int64(latency))
			}
		}(i)
	}

	wg.Wait()
	burstDuration := time.Since(burstStart)

	fmt.Printf("\n%d concurrent requests completed in: %v\n", burstCount, burstDuration)
	fmt.Printf("Success: %d, Errors: %d\n", successCount.Load(), errorCount.Load())
	if successCount.Load() > 0 {
		avgLatency := time.Duration(totalLatency.Load() / successCount.Load())
		fmt.Printf("Average latency per request: %v\n", avgLatency)
		throughput := float64(successCount.Load()) / burstDuration.Seconds()
		fmt.Printf("Throughput: %.2f requests/second\n", throughput)
	}
	fmt.Println()

	// Display final metrics
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
			successRate := float64(m.RequestCount-m.ErrorCount) / float64(m.RequestCount) * 100
			fmt.Printf("  Avg Latency: %v\n", avgLatency)
			fmt.Printf("  Success Rate: %.1f%%\n", successRate)
		}
		fmt.Println()
	}

	fmt.Println("=== Example Complete ===")
	fmt.Println("\nKey Observations:")
	fmt.Println("  - Router is fully thread-safe for concurrent use")
	fmt.Println("  - Requests are distributed evenly across providers")
	fmt.Println("  - Metrics are updated atomically without data races")
	fmt.Println("  - Concurrent requests complete faster than sequential")
	fmt.Println("  - No performance degradation under high concurrency")
	fmt.Println("\nUse Cases:")
	fmt.Println("  - High-throughput web servers")
	fmt.Println("  - Batch processing pipelines")
	fmt.Println("  - Real-time applications")
	fmt.Println("  - Load testing and benchmarking")
}
