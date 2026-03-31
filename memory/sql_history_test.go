package memory

import (
	"context"
	"database/sql"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/LucaLanziani/langchain-go/core"
)

// newMockDB is a helper that creates a go-sqlmock database for tests.
// The CREATE TABLE statement is expected once during NewSQLHistory.
func newMockDB(t *testing.T) (*sql.DB, sqlmock.Sqlmock) {
	t.Helper()
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock.New: %v", err)
	}
	// Expect the CREATE TABLE statement from NewSQLHistory.
	mock.ExpectExec(`CREATE TABLE IF NOT EXISTS`).WillReturnResult(sqlmock.NewResult(0, 0))
	return db, mock
}

func TestSQLHistoryAddMessage(t *testing.T) {
	ctx := context.Background()
	db, mock := newMockDB(t)
	defer db.Close()

	mock.ExpectExec(`INSERT INTO chat_messages`).
		WithArgs("sess1", "human", "hello", "", sqlmock.AnyArg(), sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(1, 1))

	h, err := NewSQLHistory(db, WithSessionID("sess1"))
	if err != nil {
		t.Fatalf("NewSQLHistory: %v", err)
	}
	h.AddMessage(ctx, core.NewHumanMessage("hello"))

	msgs := h.GetMessages(ctx)
	if len(msgs) != 1 || msgs[0].GetContent() != "hello" {
		t.Errorf("unexpected messages: %v", msgs)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Errorf("unmet expectations: %v", err)
	}
}

func TestSQLHistoryLoad(t *testing.T) {
	ctx := context.Background()
	db, mock := newMockDB(t)
	defer db.Close()

	rows := sqlmock.NewRows([]string{"role", "content", "name", "tool_calls", "metadata"}).
		AddRow("human", "hi", "", nil, nil).
		AddRow("ai", "hello", "", nil, nil)
	mock.ExpectQuery(`SELECT role, content, name, tool_calls, metadata FROM chat_messages WHERE session_id`).
		WithArgs("sess2").
		WillReturnRows(rows)

	h, err := NewSQLHistory(db, WithSessionID("sess2"))
	if err != nil {
		t.Fatalf("NewSQLHistory: %v", err)
	}
	if err := h.Load(ctx); err != nil {
		t.Fatalf("Load: %v", err)
	}

	msgs := h.GetMessages(ctx)
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].GetType() != core.MessageTypeHuman || msgs[0].GetContent() != "hi" {
		t.Errorf("unexpected first message: %+v", msgs[0])
	}
	if msgs[1].GetType() != core.MessageTypeAI || msgs[1].GetContent() != "hello" {
		t.Errorf("unexpected second message: %+v", msgs[1])
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Errorf("unmet expectations: %v", err)
	}
}

func TestSQLHistoryClear(t *testing.T) {
	ctx := context.Background()
	db, mock := newMockDB(t)
	defer db.Close()

	mock.ExpectExec(`INSERT INTO chat_messages`).WillReturnResult(sqlmock.NewResult(1, 1))
	mock.ExpectExec(`DELETE FROM chat_messages WHERE session_id`).
		WithArgs("sess3").
		WillReturnResult(sqlmock.NewResult(0, 1))

	h, err := NewSQLHistory(db, WithSessionID("sess3"))
	if err != nil {
		t.Fatalf("NewSQLHistory: %v", err)
	}
	h.AddMessage(ctx, core.NewHumanMessage("delete me"))
	h.Clear(ctx)

	if msgs := h.GetMessages(ctx); len(msgs) != 0 {
		t.Errorf("expected 0 messages after clear, got %d", len(msgs))
	}
	if err := h.Err(); err != nil {
		t.Errorf("unexpected error after successful clear: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Errorf("unmet expectations: %v", err)
	}
}

func TestSQLHistoryListSessions(t *testing.T) {
	ctx := context.Background()
	db, mock := newMockDB(t)
	defer db.Close()

	rows := sqlmock.NewRows([]string{"session_id"}).
		AddRow("alpha").
		AddRow("beta").
		AddRow("gamma")
	mock.ExpectQuery(`SELECT DISTINCT session_id FROM chat_messages`).
		WillReturnRows(rows)

	h, err := NewSQLHistory(db, WithSessionID("any"))
	if err != nil {
		t.Fatalf("NewSQLHistory: %v", err)
	}
	sessions, err := h.ListSessions(ctx)
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(sessions) != 3 {
		t.Errorf("expected 3 sessions, got %d: %v", len(sessions), sessions)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Errorf("unmet expectations: %v", err)
	}
}

func TestSQLHistoryDeleteSession(t *testing.T) {
	ctx := context.Background()
	db, mock := newMockDB(t)
	defer db.Close()

	mock.ExpectExec(`DELETE FROM chat_messages WHERE session_id`).
		WithArgs("old-session").
		WillReturnResult(sqlmock.NewResult(0, 5))

	h, err := NewSQLHistory(db, WithSessionID("current"))
	if err != nil {
		t.Fatalf("NewSQLHistory: %v", err)
	}
	if err := h.DeleteSession(ctx, "old-session"); err != nil {
		t.Fatalf("DeleteSession: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Errorf("unmet expectations: %v", err)
	}
}

func TestSQLHistorySaveIsNoOp(t *testing.T) {
	ctx := context.Background()
	db, mock := newMockDB(t)
	defer db.Close()

	h, err := NewSQLHistory(db, WithSessionID("noop"))
	if err != nil {
		t.Fatalf("NewSQLHistory: %v", err)
	}
	// Save should be a no-op; no additional DB calls expected.
	if err := h.Save(ctx); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Errorf("unmet expectations: %v", err)
	}
}

func TestSQLHistoryWithCustomTableName(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock.New: %v", err)
	}
	defer db.Close()

	mock.ExpectExec(`CREATE TABLE IF NOT EXISTS my_table`).
		WillReturnResult(sqlmock.NewResult(0, 0))

	_, err = NewSQLHistory(db, WithSessionID("s"), WithTableName("my_table"))
	if err != nil {
		t.Fatalf("NewSQLHistory with custom table: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Errorf("unmet expectations: %v", err)
	}
}

func TestSQLHistoryWithBufferMemory(t *testing.T) {
	ctx := context.Background()
	db, mock := newMockDB(t)
	defer db.Close()

	// SaveContext calls AddMessage twice (human + ai), each does an INSERT.
	mock.ExpectExec(`INSERT INTO chat_messages`).WillReturnResult(sqlmock.NewResult(1, 1))
	mock.ExpectExec(`INSERT INTO chat_messages`).WillReturnResult(sqlmock.NewResult(2, 1))

	// LoadMemoryVariables calls Load which does a SELECT.
	loadRows := sqlmock.NewRows([]string{"role", "content", "name", "tool_calls", "metadata"}).
		AddRow("human", "hi", "", nil, nil).
		AddRow("ai", "hello", "", nil, nil)
	mock.ExpectQuery(`SELECT role, content, name, tool_calls, metadata FROM chat_messages`).
		WithArgs("sess-buf").
		WillReturnRows(loadRows)

	h, err := NewSQLHistory(db, WithSessionID("sess-buf"))
	if err != nil {
		t.Fatalf("NewSQLHistory: %v", err)
	}

	mem := NewConversationBufferMemory(WithChatHistory(h))
	mem.ReturnMessages = true
	if err := mem.SaveContext(ctx,
		map[string]any{"input": "hi"},
		map[string]any{"output": "hello"},
	); err != nil {
		t.Fatalf("SaveContext: %v", err)
	}

	vars, err := mem.LoadMemoryVariables(ctx, nil)
	if err != nil {
		t.Fatalf("LoadMemoryVariables: %v", err)
	}
	msgs, ok := vars["history"].([]core.Message)
	if !ok {
		t.Fatal("expected []core.Message")
	}
	if len(msgs) != 2 {
		t.Errorf("expected 2 messages, got %d", len(msgs))
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Errorf("unmet expectations: %v", err)
	}
}
