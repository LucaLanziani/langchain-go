package agents

import (
	"context"
	"errors"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
	"github.com/LucaLanziani/langchain-go/prompts"
	"github.com/LucaLanziani/langchain-go/tools"
)

// mockTool implements tools.Tool interface for testing.
type mockTool struct {
	name   string
	desc   string
	result string
	err    error
}

func (t *mockTool) Name() string                                    { return t.name }
func (t *mockTool) Description() string                              { return t.desc }
func (t *mockTool) ArgsSchema() map[string]any                       { return map[string]any{} }
func (t *mockTool) Run(_ context.Context, _ string) (string, error) { return t.result, t.err }

// mockAgent implements the Agent interface for testing.
type mockAgent struct {
	planResponses []*AgentOutput
	planErrors    []error
	callCount     int
}

func (a *mockAgent) Plan(_ context.Context, _ []AgentStep, _ map[string]any) (*AgentOutput, error) {
	idx := a.callCount
	a.callCount++
	if idx < len(a.planErrors) && a.planErrors[idx] != nil {
		return nil, a.planErrors[idx]
	}
	if idx < len(a.planResponses) {
		return a.planResponses[idx], nil
	}
	return &AgentOutput{Finish: &AgentFinish{ReturnValues: map[string]any{"output": "done"}}}, nil
}

func (a *mockAgent) InputKeys() []string  { return []string{"input"} }
func (a *mockAgent) OutputKeys() []string { return []string{"output"} }

// --- AgentExecutor tests ---

func TestNewAgentExecutorNilPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for nil agent")
		}
	}()
	NewAgentExecutor(nil, nil)
}

func TestAgentExecutorGetName(t *testing.T) {
	agent := &mockAgent{}
	exec := NewAgentExecutor(agent, nil)
	if exec.GetName() != "AgentExecutor" {
		t.Errorf("expected 'AgentExecutor', got %q", exec.GetName())
	}
	exec.name = "Custom"
	if exec.GetName() != "Custom" {
		t.Errorf("expected 'Custom', got %q", exec.GetName())
	}
}

func TestAgentExecutorOptions(t *testing.T) {
	agent := &mockAgent{}
	exec := NewAgentExecutor(agent, nil,
		WithMaxIterations(5),
		WithReturnIntermediateSteps(true),
		WithHandleParsingErrors(true),
	)
	if exec.maxIterations != 5 {
		t.Errorf("expected maxIterations=5, got %d", exec.maxIterations)
	}
	if !exec.returnIntermediateSteps {
		t.Error("expected returnIntermediateSteps=true")
	}
	if !exec.handleParsingErrors {
		t.Error("expected handleParsingErrors=true")
	}
}

func TestAgentExecutorFinishImmediately(t *testing.T) {
	agent := &mockAgent{
		planResponses: []*AgentOutput{
			{Finish: &AgentFinish{ReturnValues: map[string]any{"output": "42"}}},
		},
	}
	exec := NewAgentExecutor(agent, nil)
	result, err := exec.Invoke(context.Background(), map[string]any{"input": "what?"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["output"] != "42" {
		t.Errorf("expected output='42', got %v", result["output"])
	}
}

func TestAgentExecutorFinishWithIntermediateSteps(t *testing.T) {
	agent := &mockAgent{
		planResponses: []*AgentOutput{
			{Finish: &AgentFinish{ReturnValues: map[string]any{"output": "done"}}},
		},
	}
	exec := NewAgentExecutor(agent, nil, WithReturnIntermediateSteps(true))
	result, err := exec.Invoke(context.Background(), map[string]any{"input": "q"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := result["intermediate_steps"]; !ok {
		t.Error("expected 'intermediate_steps' in result")
	}
}

func TestAgentExecutorCallsToolThenFinishes(t *testing.T) {
	tool := &mockTool{name: "calculator", desc: "calculates", result: "4"}
	agent := &mockAgent{
		planResponses: []*AgentOutput{
			{Actions: []AgentAction{{Tool: "calculator", ToolInput: "2+2"}}},
			{Finish: &AgentFinish{ReturnValues: map[string]any{"output": "4"}}},
		},
	}
	exec := NewAgentExecutor(agent, []tools.Tool{tool})
	result, err := exec.Invoke(context.Background(), map[string]any{"input": "2+2"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["output"] != "4" {
		t.Errorf("expected output='4', got %v", result["output"])
	}
}

func TestAgentExecutorToolError(t *testing.T) {
	tool := &mockTool{name: "errtool", desc: "errors", err: errors.New("tool failed")}
	agent := &mockAgent{
		planResponses: []*AgentOutput{
			{Actions: []AgentAction{{Tool: "errtool", ToolInput: "x"}}},
			{Finish: &AgentFinish{ReturnValues: map[string]any{"output": "recovered"}}},
		},
	}
	exec := NewAgentExecutor(agent, []tools.Tool{tool})
	result, err := exec.Invoke(context.Background(), map[string]any{"input": "q"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["output"] != "recovered" {
		t.Errorf("expected output='recovered', got %v", result["output"])
	}
}

func TestAgentExecutorUnknownTool(t *testing.T) {
	// With a known tool list so availableToolNames exercises the loop
	knownTool := &mockTool{name: "known_tool", desc: "a known tool", result: "ok"}
	agent := &mockAgent{
		planResponses: []*AgentOutput{
			{Actions: []AgentAction{{Tool: "unknown_tool", ToolInput: "x"}}},
			{Finish: &AgentFinish{ReturnValues: map[string]any{"output": "ok"}}},
		},
	}
	exec := NewAgentExecutor(agent, []tools.Tool{knownTool})
	result, err := exec.Invoke(context.Background(), map[string]any{"input": "q"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["output"] != "ok" {
		t.Errorf("expected output='ok', got %v", result["output"])
	}
}

func TestAgentExecutorMaxIterations(t *testing.T) {
	// Agent always returns an action, never finishes.
	agent := &mockAgent{}
	// Add many identical action responses
	for i := 0; i < 20; i++ {
		agent.planResponses = append(agent.planResponses,
			&AgentOutput{Actions: []AgentAction{{Tool: "non_existent", ToolInput: "x"}}})
	}
	exec := NewAgentExecutor(agent, nil, WithMaxIterations(3))
	_, err := exec.Invoke(context.Background(), map[string]any{"input": "q"})
	if err == nil {
		t.Error("expected error for max iterations exceeded")
	}
}

func TestAgentExecutorPlanError(t *testing.T) {
	agent := &mockAgent{
		planErrors: []error{errors.New("planning failed")},
	}
	exec := NewAgentExecutor(agent, nil)
	_, err := exec.Invoke(context.Background(), map[string]any{"input": "q"})
	if err == nil {
		t.Error("expected error from planning failure")
	}
}

func TestAgentExecutorHandleParsingErrors(t *testing.T) {
	agent := &mockAgent{
		planErrors: []error{
			errors.New("parse error"),
			nil,
		},
		planResponses: []*AgentOutput{
			nil, // first response is nil due to error
			{Finish: &AgentFinish{ReturnValues: map[string]any{"output": "retried"}}},
		},
	}
	exec := NewAgentExecutor(agent, nil, WithHandleParsingErrors(true))
	result, err := exec.Invoke(context.Background(), map[string]any{"input": "q"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["output"] != "retried" {
		t.Errorf("expected output='retried', got %v", result["output"])
	}
}

func TestAgentExecutorContextCancelled(t *testing.T) {
	agent := &mockAgent{}
	for i := 0; i < 5; i++ {
		agent.planResponses = append(agent.planResponses,
			&AgentOutput{Actions: []AgentAction{{Tool: "non_existent", ToolInput: "x"}}})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately
	exec := NewAgentExecutor(agent, nil)
	_, err := exec.Invoke(ctx, map[string]any{"input": "q"})
	if err == nil {
		t.Error("expected error from cancelled context")
	}
}

func TestAgentExecutorWithCallbacks(t *testing.T) {
	var started, agentFinished, ended bool
	cb := &testAgentCallback{
		onStart:       func() { started = true },
		onAgentFinish: func() { agentFinished = true },
		onEnd:         func() { ended = true },
	}
	agent := &mockAgent{
		planResponses: []*AgentOutput{
			{Finish: &AgentFinish{ReturnValues: map[string]any{"output": "done"}}},
		},
	}
	exec := NewAgentExecutor(agent, nil)
	_, err := exec.Invoke(context.Background(), map[string]any{"input": "q"}, core.WithCallbacks(cb))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !started {
		t.Error("expected OnChainStart to be called")
	}
	if !agentFinished {
		t.Error("expected OnAgentFinish to be called")
	}
	if !ended {
		t.Error("expected OnChainEnd to be called")
	}
}

func TestAgentExecutorWithCallbacksOnToolCall(t *testing.T) {
	var toolStarted, toolEnded bool
	cb := &testAgentCallback{
		onToolStart: func() { toolStarted = true },
		onToolEnd:   func() { toolEnded = true },
	}
	tool := &mockTool{name: "calc", desc: "calc", result: "4"}
	agent := &mockAgent{
		planResponses: []*AgentOutput{
			{Actions: []AgentAction{{Tool: "calc", ToolInput: "2+2"}}},
			{Finish: &AgentFinish{ReturnValues: map[string]any{"output": "ok"}}},
		},
	}
	exec := NewAgentExecutor(agent, []tools.Tool{tool})
	_, _ = exec.Invoke(context.Background(), map[string]any{"input": "q"}, core.WithCallbacks(cb))
	if !toolStarted {
		t.Error("expected OnToolStart to be called")
	}
	if !toolEnded {
		t.Error("expected OnToolEnd to be called")
	}
}

func TestAgentExecutorWithCallbacksOnToolError(t *testing.T) {
	var toolErrored bool
	cb := &testAgentCallback{
		onToolError: func() { toolErrored = true },
	}
	tool := &mockTool{name: "bad", desc: "bad", err: errors.New("tool error")}
	agent := &mockAgent{
		planResponses: []*AgentOutput{
			{Actions: []AgentAction{{Tool: "bad", ToolInput: "x"}}},
			{Finish: &AgentFinish{ReturnValues: map[string]any{"output": "ok"}}},
		},
	}
	exec := NewAgentExecutor(agent, []tools.Tool{tool})
	_, _ = exec.Invoke(context.Background(), map[string]any{"input": "q"}, core.WithCallbacks(cb))
	if !toolErrored {
		t.Error("expected OnToolError to be called")
	}
}

func TestAgentExecutorWithCallbacksAndAgentAction(t *testing.T) {
	var actionCalled bool
	cb := &testAgentCallback{
		onAgentAction: func() { actionCalled = true },
	}
	agent := &mockAgent{
		planResponses: []*AgentOutput{
			{Actions: []AgentAction{{Tool: "non_existent", ToolInput: "x"}}},
			{Finish: &AgentFinish{ReturnValues: map[string]any{"output": "ok"}}},
		},
	}
	exec := NewAgentExecutor(agent, nil)
	_, _ = exec.Invoke(context.Background(), map[string]any{"input": "q"}, core.WithCallbacks(cb))
	if !actionCalled {
		t.Error("expected OnAgentAction to be called")
	}
}

func TestAgentExecutorPlanErrorWithCallback(t *testing.T) {
	var errored bool
	cb := &testAgentCallback{onError: func() { errored = true }}
	agent := &mockAgent{
		planErrors: []error{errors.New("plan error")},
	}
	exec := NewAgentExecutor(agent, nil)
	_, _ = exec.Invoke(context.Background(), map[string]any{"input": "q"}, core.WithCallbacks(cb))
	if !errored {
		t.Error("expected OnChainError to be called")
	}
}

func TestAgentExecutorMaxIterationsWithCallback(t *testing.T) {
	var errored bool
	cb := &testAgentCallback{onError: func() { errored = true }}
	agent := &mockAgent{}
	for i := 0; i < 5; i++ {
		agent.planResponses = append(agent.planResponses,
			&AgentOutput{Actions: []AgentAction{{Tool: "non_existent", ToolInput: "x"}}})
	}
	exec := NewAgentExecutor(agent, nil, WithMaxIterations(2))
	_, _ = exec.Invoke(context.Background(), map[string]any{"input": "q"}, core.WithCallbacks(cb))
	if !errored {
		t.Error("expected OnChainError to be called on max iterations")
	}
}

func TestAgentExecutorStream(t *testing.T) {
	agent := &mockAgent{
		planResponses: []*AgentOutput{
			{Finish: &AgentFinish{ReturnValues: map[string]any{"output": "streamed"}}},
		},
	}
	exec := NewAgentExecutor(agent, nil)
	iter, err := exec.Stream(context.Background(), map[string]any{"input": "q"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	chunk, ok, err := iter.Next()
	if err != nil || !ok {
		t.Fatalf("expected chunk, got ok=%v err=%v", ok, err)
	}
	if chunk["output"] != "streamed" {
		t.Errorf("expected output='streamed', got %v", chunk["output"])
	}
}

func TestAgentExecutorStreamError(t *testing.T) {
	agent := &mockAgent{
		planErrors: []error{errors.New("stream plan error")},
	}
	exec := NewAgentExecutor(agent, nil)
	_, err := exec.Stream(context.Background(), map[string]any{"input": "q"})
	if err == nil {
		t.Error("expected error from stream")
	}
}

func TestAgentExecutorBatch(t *testing.T) {
	agent := &mockAgent{
		planResponses: []*AgentOutput{
			{Finish: &AgentFinish{ReturnValues: map[string]any{"output": "a"}}},
			{Finish: &AgentFinish{ReturnValues: map[string]any{"output": "b"}}},
		},
	}
	exec := NewAgentExecutor(agent, nil)
	results, err := exec.Batch(context.Background(), []map[string]any{
		{"input": "q1"},
		{"input": "q2"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
}

func TestAgentExecutorBatchError(t *testing.T) {
	agent := &mockAgent{
		planErrors: []error{errors.New("batch error")},
	}
	exec := NewAgentExecutor(agent, nil)
	_, err := exec.Batch(context.Background(), []map[string]any{{"input": "q"}})
	if err == nil {
		t.Error("expected error in batch")
	}
}

// testAgentCallback is a test callback handler for agent tests.
type testAgentCallback struct {
	core.BaseCallbackHandler
	onStart       func()
	onEnd         func()
	onError       func()
	onAgentFinish func()
	onAgentAction func()
	onToolStart   func()
	onToolEnd     func()
	onToolError   func()
}

func (c *testAgentCallback) OnChainStart(_ context.Context, _ map[string]any, _ string, _ string, _ map[string]any) {
	if c.onStart != nil {
		c.onStart()
	}
}

func (c *testAgentCallback) OnChainEnd(_ context.Context, _ map[string]any, _ string) {
	if c.onEnd != nil {
		c.onEnd()
	}
}

func (c *testAgentCallback) OnChainError(_ context.Context, _ error, _ string) {
	if c.onError != nil {
		c.onError()
	}
}

func (c *testAgentCallback) OnAgentFinish(_ context.Context, _ core.AgentFinishData, _ string) {
	if c.onAgentFinish != nil {
		c.onAgentFinish()
	}
}

func (c *testAgentCallback) OnAgentAction(_ context.Context, _ core.AgentActionData, _ string) {
	if c.onAgentAction != nil {
		c.onAgentAction()
	}
}

func (c *testAgentCallback) OnToolStart(_ context.Context, _ string, _ string, _ string, _ string) {
	if c.onToolStart != nil {
		c.onToolStart()
	}
}

func (c *testAgentCallback) OnToolEnd(_ context.Context, _ string, _ string) {
	if c.onToolEnd != nil {
		c.onToolEnd()
	}
}

func (c *testAgentCallback) OnToolError(_ context.Context, _ error, _ string) {
	if c.onToolError != nil {
		c.onToolError()
	}
}

// --- ReActAgent tests ---

func TestNewReActAgentDefaultPrompt(t *testing.T) {
	llm := &mockAgentLLM{}
	a := NewReActAgent(llm, nil, nil)
	if a == nil {
		t.Fatal("expected non-nil ReActAgent")
	}
}

func TestNewReActAgentCustomPrompt(t *testing.T) {
	llm := &mockAgentLLM{}
	p := prompts.NewChatPromptTemplate(
		prompts.System("You are an agent"),
		prompts.Placeholder("agent_scratchpad"),
		prompts.Human("{input}"),
	)
	a := NewReActAgent(llm, nil, p)
	if a.prompt != p {
		t.Error("expected custom prompt to be used")
	}
}

func TestReActAgentInputOutputKeys(t *testing.T) {
	llm := &mockAgentLLM{}
	a := NewReActAgent(llm, nil, nil)
	if len(a.InputKeys()) != 1 || a.InputKeys()[0] != "input" {
		t.Errorf("unexpected InputKeys: %v", a.InputKeys())
	}
	if len(a.OutputKeys()) != 1 || a.OutputKeys()[0] != "output" {
		t.Errorf("unexpected OutputKeys: %v", a.OutputKeys())
	}
}

func TestDefaultReActPrompt(t *testing.T) {
	p := DefaultReActPrompt()
	if p == nil {
		t.Fatal("expected non-nil prompt")
	}
}

func TestReActAgentPlanFinish(t *testing.T) {
	llm := &mockAgentLLM{
		content: "Thought: I know\nFinal Answer: 42",
	}
	tool := &mockTool{name: "search", desc: "search tool"}
	a := NewReActAgent(llm, []tools.Tool{tool}, nil)
	output, err := a.Plan(context.Background(), nil, map[string]any{"input": "question"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Finish == nil {
		t.Fatal("expected Finish output")
	}
	if output.Finish.ReturnValues["output"] != "42" {
		t.Errorf("expected output '42', got %v", output.Finish.ReturnValues["output"])
	}
}

func TestReActAgentPlanAction(t *testing.T) {
	llm := &mockAgentLLM{
		content: "Thought: I need to search\nAction: search\nAction Input: golang",
	}
	tool := &mockTool{name: "search", desc: "search"}
	a := NewReActAgent(llm, []tools.Tool{tool}, nil)
	output, err := a.Plan(context.Background(), nil, map[string]any{"input": "question"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(output.Actions) != 1 {
		t.Fatalf("expected 1 action, got %d", len(output.Actions))
	}
	if output.Actions[0].Tool != "search" {
		t.Errorf("expected tool 'search', got %q", output.Actions[0].Tool)
	}
}

func TestReActAgentPlanWithIntermediateSteps(t *testing.T) {
	llm := &mockAgentLLM{
		content: "Thought: Now I know\nFinal Answer: done",
	}
	a := NewReActAgent(llm, nil, nil)
	steps := []AgentStep{
		{
			Action:      AgentAction{Tool: "search", ToolInput: "x", Log: "Thought: searching\nAction: search\nAction Input: x"},
			Observation: "search result",
		},
	}
	output, err := a.Plan(context.Background(), steps, map[string]any{"input": "q"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Finish == nil {
		t.Fatal("expected finish")
	}
}

func TestReActAgentPlanLLMError(t *testing.T) {
	llm := &mockAgentLLM{err: errors.New("llm error")}
	a := NewReActAgent(llm, nil, nil)
	_, err := a.Plan(context.Background(), nil, map[string]any{"input": "q"})
	if err == nil {
		t.Error("expected error from LLM failure")
	}
}

// ReActAgent plan parse error (invalid output).
func TestReActAgentPlanParseError(t *testing.T) {
	llm := &mockAgentLLM{content: "just random text"}
	a := NewReActAgent(llm, nil, nil)
	_, err := a.Plan(context.Background(), nil, map[string]any{"input": "q"})
	if err == nil {
		t.Error("expected parse error")
	}
}

func TestReActAgentPlanPromptError(t *testing.T) {
	llm := &mockAgentLLM{content: "Final Answer: ok"}
	// Custom prompt with a placeholder that will receive a wrong type.
	customPrompt := prompts.NewChatPromptTemplate(
		prompts.Placeholder("extra_msgs"),
		prompts.Placeholder("agent_scratchpad"),
		prompts.Human("{input}"),
	)
	a := NewReActAgent(llm, nil, customPrompt)
	// Pass "extra_msgs" as an int — invalid type for a placeholder.
	_, err := a.Plan(context.Background(), nil, map[string]any{
		"input":      "q",
		"extra_msgs": 42,
	})
	if err == nil {
		t.Error("expected error from prompt format failure")
	}
}

// --- ToolCallingAgent tests ---

func TestToolCallingAgentInputOutputKeys(t *testing.T) {
	llm := &mockAgentLLM{}
	p := prompts.NewChatPromptTemplate(
		prompts.System("system"),
		prompts.Placeholder("agent_scratchpad"),
		prompts.Human("{input}"),
	)
	a := NewToolCallingAgent(llm, nil, p)
	keys := a.InputKeys()
	// Should contain "input" but not "agent_scratchpad"
	found := false
	for _, k := range keys {
		if k == "input" {
			found = true
		}
		if k == "agent_scratchpad" {
			t.Error("agent_scratchpad should be excluded from InputKeys")
		}
	}
	if !found {
		t.Error("expected 'input' in InputKeys")
	}
	if len(a.OutputKeys()) != 1 || a.OutputKeys()[0] != "output" {
		t.Errorf("unexpected OutputKeys: %v", a.OutputKeys())
	}
}

func TestToolCallingAgentPlanFinish(t *testing.T) {
	llm := &mockAgentLLM{content: "The answer is 42"}
	p := prompts.NewChatPromptTemplate(
		prompts.Human("{input}"),
		prompts.Placeholder("agent_scratchpad"),
	)
	a := NewToolCallingAgent(llm, nil, p)
	output, err := a.Plan(context.Background(), nil, map[string]any{"input": "what?"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Finish == nil {
		t.Fatal("expected Finish output")
	}
	if output.Finish.ReturnValues["output"] != "The answer is 42" {
		t.Errorf("unexpected output: %v", output.Finish.ReturnValues["output"])
	}
}

func TestToolCallingAgentPlanWithToolCalls(t *testing.T) {
	llm := &mockAgentLLM{
		toolCalls: []core.ToolCall{
			{ID: "call_1", Name: "calculator", Args: []byte(`{"expr":"2+2"}`)},
		},
	}
	p := prompts.NewChatPromptTemplate(
		prompts.Human("{input}"),
		prompts.Placeholder("agent_scratchpad"),
	)
	a := NewToolCallingAgent(llm, nil, p)
	output, err := a.Plan(context.Background(), nil, map[string]any{"input": "what?"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(output.Actions) != 1 {
		t.Fatalf("expected 1 action, got %d", len(output.Actions))
	}
	if output.Actions[0].Tool != "calculator" {
		t.Errorf("expected tool 'calculator', got %q", output.Actions[0].Tool)
	}
}

func TestToolCallingAgentPlanWithIntermediateSteps(t *testing.T) {
	llm := &mockAgentLLM{content: "done"}
	p := prompts.NewChatPromptTemplate(
		prompts.Human("{input}"),
		prompts.Placeholder("agent_scratchpad"),
	)
	a := NewToolCallingAgent(llm, nil, p)
	steps := []AgentStep{
		{
			Action:      AgentAction{Tool: "calc", ToolInput: `{"expr":"2+2"}`, Log: "calling calc"},
			Observation: "4",
		},
	}
	output, err := a.Plan(context.Background(), steps, map[string]any{"input": "q"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Finish == nil {
		t.Fatal("expected finish")
	}
}

func TestToolCallingAgentPlanWithInvalidJSONSteps(t *testing.T) {
	// Steps where ToolInput is not valid JSON — triggers the wrapping branch in formatToolCallingSteps
	llm := &mockAgentLLM{content: "done"}
	p := prompts.NewChatPromptTemplate(
		prompts.Human("{input}"),
		prompts.Placeholder("agent_scratchpad"),
	)
	a := NewToolCallingAgent(llm, nil, p)
	steps := []AgentStep{
		{
			Action:      AgentAction{Tool: "search", ToolInput: "plain text input (not JSON)", Log: "searching"},
			Observation: "result",
		},
	}
	output, err := a.Plan(context.Background(), steps, map[string]any{"input": "q"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Finish == nil {
		t.Fatal("expected finish")
	}
}

func TestToolCallingAgentPlanLLMError(t *testing.T) {
	llm := &mockAgentLLM{err: errors.New("llm failed")}
	p := prompts.NewChatPromptTemplate(
		prompts.Human("{input}"),
		prompts.Placeholder("agent_scratchpad"),
	)
	a := NewToolCallingAgent(llm, nil, p)
	_, err := a.Plan(context.Background(), nil, map[string]any{"input": "q"})
	if err == nil {
		t.Error("expected error from LLM failure")
	}
}

func TestToolCallingAgentPlanPromptError(t *testing.T) {
	llm := &mockAgentLLM{content: "done"}
	// Prompt with a placeholder that will receive wrong type.
	p := prompts.NewChatPromptTemplate(
		prompts.Placeholder("extra_msgs"),
		prompts.Human("{input}"),
		prompts.Placeholder("agent_scratchpad"),
	)
	a := NewToolCallingAgent(llm, nil, p)
	// Pass "extra_msgs" as an int — invalid type for a placeholder.
	_, err := a.Plan(context.Background(), nil, map[string]any{
		"input":      "q",
		"extra_msgs": 42,
	})
	if err == nil {
		t.Error("expected error from prompt format failure")
	}
}

// mockAgentLLM is a mock ChatModel for agent tests.
type mockAgentLLM struct {
	content   string
	err       error
	toolCalls []core.ToolCall
}

func (m *mockAgentLLM) Invoke(_ context.Context, _ []core.Message, _ ...core.Option) (*core.AIMessage, error) {
	if m.err != nil {
		return nil, m.err
	}
	msg := core.NewAIMessageWithToolCalls(m.content, m.toolCalls)
	return msg, nil
}

func (m *mockAgentLLM) Stream(_ context.Context, _ []core.Message, _ ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
	if m.err != nil {
		return nil, m.err
	}
	ch := make(chan core.StreamChunk[*core.AIMessage], 1)
	ch <- core.StreamChunk[*core.AIMessage]{Value: core.NewAIMessage(m.content)}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

func (m *mockAgentLLM) Batch(_ context.Context, inputs [][]core.Message, _ ...core.Option) ([]*core.AIMessage, error) {
	results := make([]*core.AIMessage, len(inputs))
	for i := range inputs {
		results[i] = core.NewAIMessage(m.content)
	}
	return results, nil
}

func (m *mockAgentLLM) Generate(_ context.Context, _ []core.Message, _ ...core.Option) (*llms.ChatResult, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &llms.ChatResult{
		Generations: []*llms.ChatGeneration{{Message: core.NewAIMessage(m.content)}},
	}, nil
}

func (m *mockAgentLLM) GetName() string                                    { return "MockAgentLLM" }
func (m *mockAgentLLM) BindTools(...llms.ToolDefinition) llms.ChatModel    { return m }
func (m *mockAgentLLM) WithStructuredOutput(map[string]any) llms.ChatModel { return m }
