// Package callbacks provides callback handler implementations for
// observability, tracing, and debugging of LangChain executions.
package callbacks

import (
	"context"

	"github.com/langchain-go/langchain-go/core"
)

// Manager coordinates multiple callback handlers, dispatching events to all of them.
type Manager struct {
	handlers          []core.CallbackHandler
	inheritableHandlers []core.CallbackHandler
	parentRunID       string
	tags              []string
	metadata          map[string]any
}

// NewManager creates a new callback Manager.
func NewManager(handlers ...core.CallbackHandler) *Manager {
	return &Manager{
		handlers: handlers,
	}
}

// WithInheritableHandlers adds handlers that will be inherited by child managers.
func (m *Manager) WithInheritableHandlers(handlers ...core.CallbackHandler) *Manager {
	m.inheritableHandlers = append(m.inheritableHandlers, handlers...)
	return m
}

// WithParentRunID sets the parent run ID.
func (m *Manager) WithParentRunID(id string) *Manager {
	m.parentRunID = id
	return m
}

// WithTags sets tags for all dispatched events.
func (m *Manager) WithTags(tags ...string) *Manager {
	m.tags = append(m.tags, tags...)
	return m
}

// WithMetadata sets metadata for all dispatched events.
func (m *Manager) WithMetadata(metadata map[string]any) *Manager {
	m.metadata = metadata
	return m
}

// GetChild creates a child manager that inherits inheritable handlers.
func (m *Manager) GetChild(tag string) *Manager {
	child := &Manager{
		handlers:    append([]core.CallbackHandler{}, m.inheritableHandlers...),
		parentRunID: "",
	}
	if tag != "" {
		child.tags = append(child.tags, tag)
	}
	return child
}

// AllHandlers returns all registered handlers.
func (m *Manager) AllHandlers() []core.CallbackHandler {
	return m.handlers
}

// OnLLMStart dispatches to all handlers.
func (m *Manager) OnLLMStart(ctx context.Context, prompts []string, runID string, parentRunID string, extras map[string]any) {
	for _, h := range m.handlers {
		h.OnLLMStart(ctx, prompts, runID, parentRunID, extras)
	}
}

// OnChatModelStart dispatches to all handlers.
func (m *Manager) OnChatModelStart(ctx context.Context, messages []core.Message, runID string, parentRunID string, extras map[string]any) {
	for _, h := range m.handlers {
		h.OnChatModelStart(ctx, messages, runID, parentRunID, extras)
	}
}

// OnLLMNewToken dispatches to all handlers.
func (m *Manager) OnLLMNewToken(ctx context.Context, token string, runID string) {
	for _, h := range m.handlers {
		h.OnLLMNewToken(ctx, token, runID)
	}
}

// OnLLMEnd dispatches to all handlers.
func (m *Manager) OnLLMEnd(ctx context.Context, output *core.LLMResult, runID string) {
	for _, h := range m.handlers {
		h.OnLLMEnd(ctx, output, runID)
	}
}

// OnLLMError dispatches to all handlers.
func (m *Manager) OnLLMError(ctx context.Context, err error, runID string) {
	for _, h := range m.handlers {
		h.OnLLMError(ctx, err, runID)
	}
}

// OnChainStart dispatches to all handlers.
func (m *Manager) OnChainStart(ctx context.Context, inputs map[string]any, runID string, parentRunID string, extras map[string]any) {
	for _, h := range m.handlers {
		h.OnChainStart(ctx, inputs, runID, parentRunID, extras)
	}
}

// OnChainEnd dispatches to all handlers.
func (m *Manager) OnChainEnd(ctx context.Context, outputs map[string]any, runID string) {
	for _, h := range m.handlers {
		h.OnChainEnd(ctx, outputs, runID)
	}
}

// OnChainError dispatches to all handlers.
func (m *Manager) OnChainError(ctx context.Context, err error, runID string) {
	for _, h := range m.handlers {
		h.OnChainError(ctx, err, runID)
	}
}

// OnToolStart dispatches to all handlers.
func (m *Manager) OnToolStart(ctx context.Context, toolName string, input string, runID string, parentRunID string) {
	for _, h := range m.handlers {
		h.OnToolStart(ctx, toolName, input, runID, parentRunID)
	}
}

// OnToolEnd dispatches to all handlers.
func (m *Manager) OnToolEnd(ctx context.Context, output string, runID string) {
	for _, h := range m.handlers {
		h.OnToolEnd(ctx, output, runID)
	}
}

// OnToolError dispatches to all handlers.
func (m *Manager) OnToolError(ctx context.Context, err error, runID string) {
	for _, h := range m.handlers {
		h.OnToolError(ctx, err, runID)
	}
}

// OnAgentAction dispatches to all handlers.
func (m *Manager) OnAgentAction(ctx context.Context, action core.AgentActionData, runID string) {
	for _, h := range m.handlers {
		h.OnAgentAction(ctx, action, runID)
	}
}

// OnAgentFinish dispatches to all handlers.
func (m *Manager) OnAgentFinish(ctx context.Context, finish core.AgentFinishData, runID string) {
	for _, h := range m.handlers {
		h.OnAgentFinish(ctx, finish, runID)
	}
}

// OnRetrieverStart dispatches to all handlers.
func (m *Manager) OnRetrieverStart(ctx context.Context, query string, runID string, parentRunID string) {
	for _, h := range m.handlers {
		h.OnRetrieverStart(ctx, query, runID, parentRunID)
	}
}

// OnRetrieverEnd dispatches to all handlers.
func (m *Manager) OnRetrieverEnd(ctx context.Context, documents []*core.Document, runID string) {
	for _, h := range m.handlers {
		h.OnRetrieverEnd(ctx, documents, runID)
	}
}

// OnRetrieverError dispatches to all handlers.
func (m *Manager) OnRetrieverError(ctx context.Context, err error, runID string) {
	for _, h := range m.handlers {
		h.OnRetrieverError(ctx, err, runID)
	}
}

// OnText dispatches to all handlers.
func (m *Manager) OnText(ctx context.Context, text string, runID string) {
	for _, h := range m.handlers {
		h.OnText(ctx, text, runID)
	}
}

// Ensure Manager implements CallbackHandler.
var _ core.CallbackHandler = (*Manager)(nil)
