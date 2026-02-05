// Example: Agent with tool use — demonstrates the "20 lines of Go" goal.
//
// This creates a tool-calling agent that can use a calculator and a
// search tool to answer questions.
//
// Usage:
//
//	export OPENAI_API_KEY=sk-...
//	go run ./examples/agent_tool_use/
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/LucaLanziani/langchain-go/agents"
	"github.com/LucaLanziani/langchain-go/callbacks"
	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/prompts"
	"github.com/LucaLanziani/langchain-go/providers/openai"
	"github.com/LucaLanziani/langchain-go/tools"
)

func main() {
	ctx := context.Background()

	// Define tools.
	calculator := tools.NewTool(
		"calculator",
		"Useful for math calculations. Input should be a math expression.",
		func(_ context.Context, input string) (string, error) {
			// In a real app, you'd evaluate the expression.
			return "42", nil
		},
	)

	search := tools.NewTool(
		"search",
		"Search the web for current information. Input should be a search query.",
		func(_ context.Context, input string) (string, error) {
			return fmt.Sprintf("Search results for '%s': Go is an open-source programming language created at Google.", input), nil
		},
	)

	agentTools := []tools.Tool{calculator, search}

	// Create the prompt.
	prompt := prompts.NewChatPromptTemplate(
		prompts.System("You are a helpful assistant. Use tools when needed to answer questions accurately."),
		prompts.Placeholder("agent_scratchpad"),
		prompts.Human("{input}"),
	)

	// Create the agent and executor.
	model := openai.New()
	agent := agents.NewToolCallingAgent(model, agentTools, prompt)
	executor := agents.NewAgentExecutor(agent, agentTools,
		agents.WithMaxIterations(5),
	)

	// Run with a stdout callback for visibility.
	result, err := executor.Invoke(ctx, map[string]any{
		"input": "What is Go programming language? Also, what is 6 * 7?",
	}, core.WithCallbacks(callbacks.NewStdoutHandler()))
	if err != nil {
		log.Fatalf("Error: %v", err)
	}

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Printf("Final Answer: %s\n", result["output"])
}
