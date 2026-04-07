package main

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/provider"
)

// Example 11: Multiple Instances of Same Provider
// Demonstrates creating multiple instances of the same provider with different
// configurations (e.g., different models, temperatures) and routing between them.

func main() {
	ctx := context.Background()

	fmt.Println("=== Creating Router with Multiple OpenAI Instances ===\n")

	// Create router with three different OpenAI configurations
	router, err := provider.NewRouter(
		ctx,
		[]provider.ProviderEntry{
			{
				Name:         "openai-fast",
				ProviderType: provider.ProviderOpenAI,
				Options: []provider.ProviderOption{
					provider.WithModel("gpt-3.5-turbo"),
					provider.WithTemperature(0.3), // Low temperature for consistency
					provider.WithMaxTokens(500),
				},
				Tags: []string{"fast", "cheap", "precise"},
			},
			{
				Name:         "openai-smart",
				ProviderType: provider.ProviderOpenAI,
				Options: []provider.ProviderOption{
					provider.WithModel("gpt-4o"),
					provider.WithTemperature(0.7), // Moderate temperature
					provider.WithMaxTokens(2000),
				},
				Tags: []string{"smart", "expensive", "capable"},
			},
			{
				Name:         "openai-creative",
				ProviderType: provider.ProviderOpenAI,
				Options: []provider.ProviderOption{
					provider.WithModel("gpt-4o"),
					provider.WithTemperature(1.2), // High temperature for creativity
					provider.WithMaxTokens(1500),
				},
				Tags: []string{"creative", "diverse"},
			},
		},
		// Use rule-based routing to select appropriate instance
		provider.NewRuleBasedStrategy(
			[]provider.RoutingRule{
				{
					Name:     "creative-tasks",
					Priority: 100,
					Condition: func(reqCtx provider.RequestContext) bool {
						// Check message content for creative keywords
						for _, msg := range reqCtx.Messages {
							content := strings.ToLower(msg.GetContent())
							if strings.Contains(content, "story") ||
								strings.Contains(content, "creative") ||
								strings.Contains(content, "brainstorm") ||
								strings.Contains(content, "ideas") {
								return true
							}
						}
						return false
					},
					Provider: "openai-creative",
				},
				{
					Name:     "complex-tasks",
					Priority: 90,
					Condition: func(reqCtx provider.RequestContext) bool {
						// Route complex requests to smart instance
						return reqCtx.Complexity == "complex" || reqCtx.TotalTokens > 3000
					},
					Provider: "openai-smart",
				},
				{
					Name:     "simple-tasks",
					Priority: 80,
					Condition: func(reqCtx provider.RequestContext) bool {
						// Route simple requests to fast instance
						return reqCtx.Complexity == "simple"
					},
					Provider: "openai-fast",
				},
			},
			"openai-fast", // default to fast instance
		),
	)
	if err != nil {
		log.Fatalf("Failed to create router: %v", err)
	}
	defer router.Cleanup()

	fmt.Println("Router created with 3 OpenAI instances:")
	fmt.Println("  1. openai-fast (gpt-3.5-turbo, temp=0.3)")
	fmt.Println("     → Fast and precise for simple queries")
	fmt.Println("  2. openai-smart (gpt-4o, temp=0.7)")
	fmt.Println("     → Capable and balanced for complex tasks")
	fmt.Println("  3. openai-creative (gpt-4o, temp=1.2)")
	fmt.Println("     → Creative and diverse for writing tasks")
	fmt.Println()

	// Test 1: Simple factual question (should use openai-fast)
	fmt.Println("=== Test 1: Simple Factual Question ===")
	fmt.Println("Question: What is the capital of Japan?")
	response, err := router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("What is the capital of Japan?"),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Test 2: Complex analysis (should use openai-smart)
	fmt.Println("=== Test 2: Complex Analysis ===")
	complexQuestion := `Analyze the economic implications of artificial intelligence 
on the job market over the next decade, considering both displacement and creation 
of new job categories. Provide specific examples and data-driven insights.`
	fmt.Printf("Question: %s\n", complexQuestion)
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage(complexQuestion),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Test 3: Creative writing (should use openai-creative)
	fmt.Println("=== Test 3: Creative Writing ===")
	fmt.Println("Request: Write a short story about a robot learning to paint")
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("Write a short story about a robot learning to paint"),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Test 4: Another simple question (should use openai-fast)
	fmt.Println("=== Test 4: Another Simple Question ===")
	fmt.Println("Question: What is 15 * 8?")
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("What is 15 * 8?"),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Test 5: Brainstorming (should use openai-creative)
	fmt.Println("=== Test 5: Brainstorming ===")
	fmt.Println("Request: Generate 5 unique startup ideas for sustainable technology")
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("Generate 5 unique startup ideas for sustainable technology"),
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
	fmt.Println("\nKey Benefits:")
	fmt.Println("  - Cost optimization: Use cheaper models for simple tasks")
	fmt.Println("  - Quality optimization: Use better models for complex tasks")
	fmt.Println("  - Creativity control: Adjust temperature per use case")
	fmt.Println("  - Independent metrics: Track performance per configuration")
}
