// Example: Streaming chat completion with OpenAI.
//
// Usage:
//
//	export OPENAI_API_KEY=sk-...
//	go run ./examples/streaming/
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/providers/openai"
)

func main() {
	ctx := context.Background()

	model := openai.New()

	messages := []core.Message{
		core.NewSystemMessage("You are a concise assistant."),
		core.NewHumanMessage("Write a haiku about Go programming."),
	}

	// Stream the response token by token.
	stream, err := model.Stream(ctx, messages)
	if err != nil {
		log.Fatalf("Error: %v", err)
	}

	fmt.Print("Response: ")
	for {
		chunk, ok, err := stream.Next()
		if err != nil {
			log.Fatalf("Stream error: %v", err)
		}
		if !ok {
			break
		}
		fmt.Print(chunk.Content)
	}
	fmt.Println()
}
