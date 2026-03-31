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
	prompt := NewChatPromptTemplate(
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

func TestChatPromptTemplateWithName(t *testing.T) {
	prompt := NewChatPromptTemplate(Human("hi"))
	if prompt.GetName() != "ChatPromptTemplate" {
		t.Errorf("expected default name 'ChatPromptTemplate', got %q", prompt.GetName())
	}
	prompt.WithName("MyPrompt")
	if prompt.GetName() != "MyPrompt" {
		t.Errorf("expected 'MyPrompt', got %q", prompt.GetName())
	}
}

func TestChatPromptTemplateWithPartialVariables(t *testing.T) {
	prompt := NewChatPromptTemplate(Human("{greeting}, {name}!"))
	prompt.WithPartialVariables(map[string]any{"greeting": "Hello"})

	msgs, err := prompt.FormatMessages(map[string]any{"name": "World"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) != 1 || msgs[0].GetContent() != "Hello, World!" {
		t.Errorf("unexpected content: %q", msgs[0].GetContent())
	}
}

func TestChatPromptTemplateAIMessage(t *testing.T) {
	prompt := NewChatPromptTemplate(AI("I am {bot}"))

	msgs, err := prompt.FormatMessages(map[string]any{"bot": "Copilot"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	if msgs[0].GetType() != core.MessageTypeAI {
		t.Errorf("expected AI message type, got %s", msgs[0].GetType())
	}
	if msgs[0].GetContent() != "I am Copilot" {
		t.Errorf("unexpected content: %q", msgs[0].GetContent())
	}
}

func TestChatPromptTemplatePlaceholderInvalidType(t *testing.T) {
	prompt := NewChatPromptTemplate(Placeholder("history"))
	_, err := prompt.FormatMessages(map[string]any{"history": "not a slice"})
	if err == nil {
		t.Error("expected error for invalid placeholder type")
	}
}

func TestChatPromptTemplateGenericRole(t *testing.T) {
	prompt := NewChatPromptTemplate(MessageTemplate{Role: "tool", Template: "result: {output}"})

	msgs, err := prompt.FormatMessages(map[string]any{"output": "done"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	if msgs[0].GetContent() != "result: done" {
		t.Errorf("unexpected content: %q", msgs[0].GetContent())
	}
}

func TestMessagesPlaceholderFunc(t *testing.T) {
	mt := Placeholder("history")
	if mt.Role != "placeholder" {
		t.Errorf("expected role 'placeholder', got %q", mt.Role)
	}
	if mt.Template != "history" {
		t.Errorf("expected template 'history', got %q", mt.Template)
	}
}

func TestChatPromptTemplateStream(t *testing.T) {
	prompt := NewChatPromptTemplate(Human("{msg}"))

	iter, err := prompt.Stream(context.Background(), map[string]any{"msg": "hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	chunk, ok, err := iter.Next()
	if err != nil {
		t.Fatalf("unexpected error from Next: %v", err)
	}
	if !ok {
		t.Fatal("expected a chunk")
	}
	if len(chunk) != 1 || chunk[0].GetContent() != "hello" {
		t.Errorf("unexpected chunk: %v", chunk)
	}

	_, ok, err = iter.Next()
	if err != nil {
		t.Fatalf("unexpected error on second Next: %v", err)
	}
	if ok {
		t.Error("expected stream to be done")
	}
}

func TestChatPromptTemplateStreamError(t *testing.T) {
	prompt := NewChatPromptTemplate(Placeholder("history"))
	_, err := prompt.Stream(context.Background(), map[string]any{"history": "not-a-slice"})
	if err == nil {
		t.Error("expected error for invalid placeholder type in stream")
	}
}

func TestChatPromptTemplateBatch(t *testing.T) {
	prompt := NewChatPromptTemplate(Human("{item}"))

	results, err := prompt.Batch(context.Background(), []map[string]any{
		{"item": "first"},
		{"item": "second"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0][0].GetContent() != "first" {
		t.Errorf("unexpected result[0]: %q", results[0][0].GetContent())
	}
	if results[1][0].GetContent() != "second" {
		t.Errorf("unexpected result[1]: %q", results[1][0].GetContent())
	}
}

func TestChatPromptTemplateBatchError(t *testing.T) {
	prompt := NewChatPromptTemplate(Placeholder("history"))
	_, err := prompt.Batch(context.Background(), []map[string]any{{"history": 42}})
	if err == nil {
		t.Error("expected error for batch with invalid placeholder type")
	}
}
