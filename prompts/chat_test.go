package prompts

import (
	"context"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
)

func TestChatPromptTemplate(t *testing.T) {
	prompt := NewChatPromptTemplate(
		System("You are a helpful assistant."),
		Human("Tell me about {topic}"),
	)

	messages, err := prompt.FormatMessages(map[string]any{"topic": "Go"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}

	if messages[0].GetType() != core.MessageTypeSystem {
		t.Errorf("expected system message, got %s", messages[0].GetType())
	}
	if messages[0].GetContent() != "You are a helpful assistant." {
		t.Errorf("unexpected system content: %q", messages[0].GetContent())
	}

	if messages[1].GetType() != core.MessageTypeHuman {
		t.Errorf("expected human message, got %s", messages[1].GetType())
	}
	if messages[1].GetContent() != "Tell me about Go" {
		t.Errorf("unexpected human content: %q", messages[1].GetContent())
	}
}

func TestChatPromptTemplateWithPlaceholder(t *testing.T) {
	prompt := NewChatPromptTemplate(
		System("You are helpful."),
		Placeholder("chat_history"),
		Human("{input}"),
	)

	history := []core.Message{
		core.NewHumanMessage("Hi"),
		core.NewAIMessage("Hello!"),
	}

	messages, err := prompt.FormatMessages(map[string]any{
		"chat_history": history,
		"input":        "How are you?",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(messages))
	}

	// System, Human(history), AI(history), Human(input)
	if messages[0].GetType() != core.MessageTypeSystem {
		t.Errorf("message 0: expected system, got %s", messages[0].GetType())
	}
	if messages[1].GetType() != core.MessageTypeHuman {
		t.Errorf("message 1: expected human, got %s", messages[1].GetType())
	}
	if messages[2].GetType() != core.MessageTypeAI {
		t.Errorf("message 2: expected ai, got %s", messages[2].GetType())
	}
	if messages[3].GetType() != core.MessageTypeHuman {
		t.Errorf("message 3: expected human, got %s", messages[3].GetType())
	}
}

func TestChatPromptTemplateInvoke(t *testing.T) {
	prompt := NewChatPromptTemplate(
		Human("{question}"),
	)

	messages, err := prompt.Invoke(context.Background(), map[string]any{"question": "Why?"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	if messages[0].GetContent() != "Why?" {
		t.Errorf("unexpected content: %q", messages[0].GetContent())
	}
}

func TestFromMessages(t *testing.T) {
	prompt := FromMessages(
		System("sys"),
		Human("hello"),
	)
	if prompt == nil {
		t.Fatal("expected non-nil prompt")
	}
	if len(prompt.Messages) != 2 {
		t.Errorf("expected 2 messages, got %d", len(prompt.Messages))
	}
}
