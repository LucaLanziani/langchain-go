package agents

import (
	"testing"
)

func TestParseReActOutputFinalAnswer(t *testing.T) {
	text := `Thought: I now know the final answer
Final Answer: The answer is 42`

	output, err := parseReActOutput(text)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Finish == nil {
		t.Fatal("expected Finish, got nil")
	}
	if output.Finish.ReturnValues["output"] != "The answer is 42" {
		t.Errorf("unexpected output: %v", output.Finish.ReturnValues["output"])
	}
}

func TestParseReActOutputAction(t *testing.T) {
	text := `Thought: I need to search for this
Action: search
Action Input: golang langchain`

	output, err := parseReActOutput(text)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(output.Actions) != 1 {
		t.Fatalf("expected 1 action, got %d", len(output.Actions))
	}
	if output.Actions[0].Tool != "search" {
		t.Errorf("expected tool 'search', got %q", output.Actions[0].Tool)
	}
	if output.Actions[0].ToolInput != "golang langchain" {
		t.Errorf("expected input 'golang langchain', got %q", output.Actions[0].ToolInput)
	}
}

func TestParseReActOutputMultilineJSON(t *testing.T) {
	text := "Thought: I should search\nAction: search\nAction Input: ```json\n{\n  \"query\": \"golang\",\n  \"limit\": 2\n}\n```"

	output, err := parseReActOutput(text)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Actions[0].ToolInput != "{\n  \"query\": \"golang\",\n  \"limit\": 2\n}" {
		t.Fatalf("unexpected multiline tool input: %q", output.Actions[0].ToolInput)
	}
}

func TestParseReActOutputFinalAnswerWithWhitespace(t *testing.T) {
	text := "Thought: done\n\nFinal Answer: The answer is 42\nwith detail"

	output, err := parseReActOutput(text)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Finish.ReturnValues["output"] != "The answer is 42\nwith detail" {
		t.Fatalf("unexpected final answer: %v", output.Finish.ReturnValues["output"])
	}
}

func TestParseReActOutputInvalid(t *testing.T) {
	text := "Just some random text without structure"
	_, err := parseReActOutput(text)
	if err == nil {
		t.Error("expected error for unparseable output")
	}
}
