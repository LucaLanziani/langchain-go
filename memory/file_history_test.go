package memory

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
)

func TestFileHistoryAddAndLoad(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	h, err := NewFileHistory(dir, WithSessionID("sess1"))
	if err != nil {
		t.Fatalf("NewFileHistory: %v", err)
	}

	h.AddMessage(ctx, core.NewHumanMessage("hello"))
	h.AddMessage(ctx, core.NewAIMessage("world"))

	// Create a fresh instance that loads from disk.
	h2, err := NewFileHistory(dir, WithSessionID("sess1"))
	if err != nil {
		t.Fatalf("NewFileHistory: %v", err)
	}
	if err := h2.Load(ctx); err != nil {
		t.Fatalf("Load: %v", err)
	}
	msgs := h2.GetMessages(ctx)
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].GetContent() != "hello" || msgs[0].GetType() != core.MessageTypeHuman {
		t.Errorf("unexpected first message: %+v", msgs[0])
	}
	if msgs[1].GetContent() != "world" || msgs[1].GetType() != core.MessageTypeAI {
		t.Errorf("unexpected second message: %+v", msgs[1])
	}
}

func TestFileHistoryMessageTypes(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	h, err := NewFileHistory(dir, WithSessionID("types"))
	if err != nil {
		t.Fatalf("NewFileHistory: %v", err)
	}

	h.AddMessage(ctx, core.NewHumanMessage("hi"))
	h.AddMessage(ctx, core.NewAIMessage("hello"))
	h.AddMessage(ctx, core.NewSystemMessage("be polite"))
	h.AddMessage(ctx, core.NewToolMessage("result", "call-1"))
	h.AddMessage(ctx, core.NewFunctionMessage("fn", "fn-result"))

	h2, err := NewFileHistory(dir, WithSessionID("types"))
	if err != nil {
		t.Fatalf("NewFileHistory: %v", err)
	}
	if err := h2.Load(ctx); err != nil {
		t.Fatalf("Load: %v", err)
	}
	msgs := h2.GetMessages(ctx)
	if len(msgs) != 5 {
		t.Fatalf("expected 5 messages, got %d", len(msgs))
	}

	types := []core.MessageType{
		core.MessageTypeHuman,
		core.MessageTypeAI,
		core.MessageTypeSystem,
		core.MessageTypeTool,
		core.MessageTypeFunction,
	}
	for i, want := range types {
		if msgs[i].GetType() != want {
			t.Errorf("msg[%d]: expected type %q, got %q", i, want, msgs[i].GetType())
		}
	}
}

func TestFileHistoryClear(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	h, err := NewFileHistory(dir, WithSessionID("clr"))
	if err != nil {
		t.Fatalf("NewFileHistory: %v", err)
	}
	h.AddMessage(ctx, core.NewHumanMessage("keep me not"))
	h.Clear(ctx)

	// Load into a new instance.
	h2, err := NewFileHistory(dir, WithSessionID("clr"))
	if err != nil {
		t.Fatalf("NewFileHistory: %v", err)
	}
	if err := h2.Load(ctx); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if msgs := h2.GetMessages(ctx); len(msgs) != 0 {
		t.Errorf("expected 0 messages after clear, got %d", len(msgs))
	}
}

func TestFileHistoryListSessions(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	for _, sid := range []string{"a", "b", "c"} {
		h, err := NewFileHistory(dir, WithSessionID(sid))
		if err != nil {
			t.Fatalf("NewFileHistory(%s): %v", sid, err)
		}
		h.AddMessage(ctx, core.NewHumanMessage("msg"))
	}

	h, _ := NewFileHistory(dir, WithSessionID("a"))
	sessions, err := h.ListSessions(ctx)
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(sessions) != 3 {
		t.Errorf("expected 3 sessions, got %d: %v", len(sessions), sessions)
	}
}

func TestFileHistoryDeleteSession(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	for _, sid := range []string{"keep", "delete-me"} {
		h, _ := NewFileHistory(dir, WithSessionID(sid))
		h.AddMessage(ctx, core.NewHumanMessage("msg"))
	}

	h, _ := NewFileHistory(dir, WithSessionID("keep"))
	if err := h.DeleteSession(ctx, "delete-me"); err != nil {
		t.Fatalf("DeleteSession: %v", err)
	}

	sessions, _ := h.ListSessions(ctx)
	for _, s := range sessions {
		if s == "delete-me" {
			t.Error("deleted session still listed")
		}
	}

	// Deleting a non-existent session should not error.
	if err := h.DeleteSession(ctx, "nonexistent"); err != nil {
		t.Errorf("expected no error for non-existent session, got: %v", err)
	}
}

func TestFileHistoryMultiSessionIsolation(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	h1, _ := NewFileHistory(dir, WithSessionID("s1"))
	h2, _ := NewFileHistory(dir, WithSessionID("s2"))

	h1.AddMessage(ctx, core.NewHumanMessage("from s1"))
	h2.AddMessage(ctx, core.NewHumanMessage("from s2"))

	// Load s1 into a fresh instance and verify it only has s1's message.
	fresh1, _ := NewFileHistory(dir, WithSessionID("s1"))
	if err := fresh1.Load(ctx); err != nil {
		t.Fatalf("Load s1: %v", err)
	}
	msgs1 := fresh1.GetMessages(ctx)
	if len(msgs1) != 1 || msgs1[0].GetContent() != "from s1" {
		t.Errorf("s1 isolation failed: %v", msgs1)
	}
}

func TestFileHistoryLoadNonExistent(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	h, _ := NewFileHistory(dir, WithSessionID("ghost"))
	if err := h.Load(ctx); err != nil {
		t.Errorf("loading non-existent session should not error, got: %v", err)
	}
	if msgs := h.GetMessages(ctx); len(msgs) != 0 {
		t.Errorf("expected 0 messages for non-existent session, got %d", len(msgs))
	}
}

func TestFileHistoryAtomicWrite(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	h, _ := NewFileHistory(dir, WithSessionID("atomic"))
	h.AddMessage(ctx, core.NewHumanMessage("atomic write test"))

	// The session file must exist after Save.
	sessionFile := filepath.Join(dir, "atomic.json")
	if _, err := os.Stat(sessionFile); err != nil {
		t.Fatalf("session file should exist: %v", err)
	}

	// No temp files should remain.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if e.Name() != "atomic.json" {
			t.Errorf("unexpected file remains: %s", e.Name())
		}
	}
}

func TestFileHistoryAutoSaveDisabled(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	h, _ := NewFileHistory(dir, WithSessionID("nosave"), WithAutoSave(false))
	h.AddMessage(ctx, core.NewHumanMessage("not saved yet"))

	// File should not exist because auto-save is off.
	sessionFile := filepath.Join(dir, "nosave.json")
	if _, err := os.Stat(sessionFile); !os.IsNotExist(err) {
		t.Error("session file should not exist when auto-save is disabled")
	}

	// Explicit Save should write the file.
	if err := h.Save(ctx); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if _, err := os.Stat(sessionFile); err != nil {
		t.Fatalf("session file should exist after explicit Save: %v", err)
	}
}

func TestFileHistoryConcurrentAccess(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	const numSessions = 10
	const msgsPerSession = 5

	var wg sync.WaitGroup
	for i := 0; i < numSessions; i++ {
		wg.Add(1)
		sid := "concurrent-session-" + string(rune('0'+i))
		go func(sessionID string) {
			defer wg.Done()
			h, err := NewFileHistory(dir, WithSessionID(sessionID))
			if err != nil {
				t.Errorf("NewFileHistory(%s): %v", sessionID, err)
				return
			}
			for j := 0; j < msgsPerSession; j++ {
				h.AddMessage(ctx, core.NewHumanMessage("msg"))
			}
		}(sid)
	}
	wg.Wait()

	sessions, err := NewFileHistory(dir, WithSessionID("x"))
	if err != nil {
		t.Fatalf("NewFileHistory: %v", err)
	}
	list, _ := sessions.ListSessions(ctx)
	if len(list) != numSessions {
		t.Errorf("expected %d sessions, got %d", numSessions, len(list))
	}
}

func TestFileHistoryWithBufferMemory(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	h, err := NewFileHistory(dir, WithSessionID("buf-mem"))
	if err != nil {
		t.Fatalf("NewFileHistory: %v", err)
	}

	mem := NewConversationBufferMemory(WithChatHistory(h))
	if err := mem.SaveContext(ctx,
		map[string]any{"input": "hi"},
		map[string]any{"output": "hello"},
	); err != nil {
		t.Fatalf("SaveContext: %v", err)
	}

	// Create a fresh history and memory pointing to the same file.
	h2, err := NewFileHistory(dir, WithSessionID("buf-mem"))
	if err != nil {
		t.Fatalf("NewFileHistory: %v", err)
	}
	mem2 := NewConversationBufferMemory(WithChatHistory(h2))
	mem2.ReturnMessages = true

	vars, err := mem2.LoadMemoryVariables(ctx, nil)
	if err != nil {
		t.Fatalf("LoadMemoryVariables: %v", err)
	}
	msgs, ok := vars["history"].([]core.Message)
	if !ok {
		t.Fatal("expected []core.Message")
	}
	if len(msgs) != 2 {
		t.Errorf("expected 2 messages after reload, got %d", len(msgs))
	}
}

func TestFileHistoryWithWindowMemory(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	h, err := NewFileHistory(dir, WithSessionID("win-mem"))
	if err != nil {
		t.Fatalf("NewFileHistory: %v", err)
	}

	mem := NewConversationWindowMemory(1, WithChatHistory(h))
	for i := 0; i < 3; i++ {
		_ = mem.SaveContext(ctx, map[string]any{"input": "q"}, map[string]any{"output": "a"})
	}

	mem.ReturnMessages = true
	vars, err := mem.LoadMemoryVariables(ctx, nil)
	if err != nil {
		t.Fatalf("LoadMemoryVariables: %v", err)
	}
	msgs, ok := vars["history"].([]core.Message)
	if !ok {
		t.Fatal("expected []core.Message")
	}
	// Window K=1 means last 2 messages (1 human + 1 AI).
	if len(msgs) != 2 {
		t.Errorf("expected 2 messages for K=1, got %d", len(msgs))
	}
}
