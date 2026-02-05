package callbacks

import (
	"context"
	"fmt"
	"strings"

	"github.com/langchain-go/langchain-go/core"
)

// StdoutHandler prints callback events to stdout for debugging.
type StdoutHandler struct {
	core.BaseCallbackHandler
	// Color enables ANSI color output.
	Color bool
}

// NewStdoutHandler creates a new StdoutHandler.
func NewStdoutHandler() *StdoutHandler {
	return &StdoutHandler{Color: true}
}

func (h *StdoutHandler) OnChainStart(_ context.Context, inputs map[string]any, runID string, _ string, extras map[string]any) {
	name := "Chain"
	if n, ok := extras["name"]; ok {
		name = fmt.Sprintf("%v", n)
	}
	h.print(colorGreen, "\n\n> Entering new %s chain...\n", name)
}

func (h *StdoutHandler) OnChainEnd(_ context.Context, outputs map[string]any, _ string) {
	h.print(colorGreen, "\n> Finished chain.\n")
}

func (h *StdoutHandler) OnChainError(_ context.Context, err error, _ string) {
	h.print(colorRed, "\n> Chain error: %v\n", err)
}

func (h *StdoutHandler) OnLLMStart(_ context.Context, prompts []string, _ string, _ string, _ map[string]any) {
	h.print(colorCyan, "\n[LLM] Prompts: %s\n", strings.Join(prompts, "\n"))
}

func (h *StdoutHandler) OnChatModelStart(_ context.Context, messages []core.Message, _ string, _ string, _ map[string]any) {
	h.print(colorCyan, "\n[ChatModel] Messages:\n")
	for _, msg := range messages {
		h.print(colorCyan, "  [%s]: %s\n", msg.GetType(), truncate(msg.GetContent(), 200))
	}
}

func (h *StdoutHandler) OnLLMNewToken(_ context.Context, token string, _ string) {
	fmt.Print(token)
}

func (h *StdoutHandler) OnLLMEnd(_ context.Context, output *core.LLMResult, _ string) {
	h.print(colorCyan, "\n[LLM] Done.\n")
}

func (h *StdoutHandler) OnLLMError(_ context.Context, err error, _ string) {
	h.print(colorRed, "\n[LLM] Error: %v\n", err)
}

func (h *StdoutHandler) OnToolStart(_ context.Context, toolName string, input string, _ string, _ string) {
	h.print(colorYellow, "\n[Tool: %s] Input: %s\n", toolName, truncate(input, 200))
}

func (h *StdoutHandler) OnToolEnd(_ context.Context, output string, _ string) {
	h.print(colorYellow, "[Tool] Output: %s\n", truncate(output, 200))
}

func (h *StdoutHandler) OnToolError(_ context.Context, err error, _ string) {
	h.print(colorRed, "[Tool] Error: %v\n", err)
}

func (h *StdoutHandler) OnAgentAction(_ context.Context, action core.AgentActionData, _ string) {
	h.print(colorBlue, "\n[Agent] Action: %s\n  Input: %s\n", action.Tool, truncate(action.ToolInput, 200))
}

func (h *StdoutHandler) OnAgentFinish(_ context.Context, finish core.AgentFinishData, _ string) {
	h.print(colorBlue, "\n[Agent] Finished: %v\n", finish.Output)
}

func (h *StdoutHandler) OnRetrieverStart(_ context.Context, query string, _ string, _ string) {
	h.print(colorMagenta, "\n[Retriever] Query: %s\n", truncate(query, 200))
}

func (h *StdoutHandler) OnRetrieverEnd(_ context.Context, documents []*core.Document, _ string) {
	h.print(colorMagenta, "[Retriever] Retrieved %d documents\n", len(documents))
}

func (h *StdoutHandler) OnText(_ context.Context, text string, _ string) {
	h.print(colorWhite, "%s", text)
}

// ANSI color codes.
const (
	colorReset   = "\033[0m"
	colorRed     = "\033[31m"
	colorGreen   = "\033[32m"
	colorYellow  = "\033[33m"
	colorBlue    = "\033[34m"
	colorMagenta = "\033[35m"
	colorCyan    = "\033[36m"
	colorWhite   = "\033[37m"
)

func (h *StdoutHandler) print(color, format string, args ...any) {
	if h.Color {
		fmt.Printf(color+format+colorReset, args...)
	} else {
		fmt.Printf(format, args...)
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
