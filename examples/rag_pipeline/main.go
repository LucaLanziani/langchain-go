// Example: RAG (Retrieval Augmented Generation) pipeline.
//
// This demonstrates loading documents, splitting them, storing in a
// vector store, retrieving relevant context, and answering questions.
//
// Usage:
//
//	export OPENAI_API_KEY=sk-...
//	go run ./examples/rag_pipeline/
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/langchain-go/langchain-go/chains"
	"github.com/langchain-go/langchain-go/core"
	"github.com/langchain-go/langchain-go/prompts"
	"github.com/langchain-go/langchain-go/providers/openai"
	"github.com/langchain-go/langchain-go/retrievers"
	"github.com/langchain-go/langchain-go/textsplitters"
	"github.com/langchain-go/langchain-go/vectorstores/inmemory"
)

func main() {
	ctx := context.Background()

	// 1. Prepare documents.
	docs := []*core.Document{
		core.NewDocument("Go was designed at Google in 2007 by Robert Griesemer, Rob Pike, and Ken Thompson. It was announced in 2009 and version 1.0 was released in 2012."),
		core.NewDocument("Go is statically typed, compiled, and syntactically similar to C. It has memory safety, garbage collection, structural typing, and CSP-style concurrency."),
		core.NewDocument("Go's standard library provides built-in support for web servers, JSON, cryptography, and more. The go tool manages dependencies, builds, and tests."),
		core.NewDocument("LangChain is a framework for building applications powered by language models. It provides abstractions for chains, agents, tools, memory, and retrieval."),
	}

	// 2. Split documents into smaller chunks.
	splitter := textsplitters.NewRecursiveCharacterTextSplitter(200, 20)
	chunks := splitter.SplitDocuments(docs)
	fmt.Printf("Split %d documents into %d chunks\n", len(docs), len(chunks))

	// 3. Create embeddings and vector store.
	embedder := openai.NewEmbeddings()
	store := inmemory.New(embedder)

	ids, err := store.AddDocuments(ctx, chunks)
	if err != nil {
		log.Fatalf("Failed to add documents: %v", err)
	}
	fmt.Printf("Stored %d document chunks\n", len(ids))

	// 4. Create a retriever.
	retriever := retrievers.NewVectorStoreRetriever(store, 2)

	// 5. Create the QA chain.
	qaPrompt := prompts.NewChatPromptTemplate(
		prompts.System("Answer the question based only on the following context:\n\n{context}"),
		prompts.Human("{query}"),
	)
	model := openai.New()
	llmChain := chains.NewLLMChain(model, qaPrompt)
	qaChain := chains.NewRetrievalQA(retriever, llmChain)

	// 6. Ask a question.
	result, err := qaChain.Invoke(ctx, map[string]any{
		"query": "When was Go created and by whom?",
	})
	if err != nil {
		log.Fatalf("Error: %v", err)
	}

	fmt.Printf("\nQuestion: When was Go created and by whom?\nAnswer: %s\n", result)
}
