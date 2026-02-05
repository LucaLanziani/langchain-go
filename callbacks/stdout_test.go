package callbacks

import (
	"testing"
)

func TestNewStdoutHandler(t *testing.T) {
	h := NewStdoutHandler()
	if h == nil {
		t.Fatal("expected non-nil handler")
	}
	if !h.Color {
		t.Error("expected Color to be true by default")
	}
}

func TestTruncate(t *testing.T) {
	tests := []struct {
		input    string
		maxLen   int
		expected string
	}{
		{"hello", 10, "hello"},
		{"hello world", 5, "hello..."},
		{"", 5, ""},
		{"abcdef", 3, "abc..."},
	}

	for _, tt := range tests {
		result := truncate(tt.input, tt.maxLen)
		if result != tt.expected {
			t.Errorf("truncate(%q, %d): expected %q, got %q", tt.input, tt.maxLen, tt.expected, result)
		}
	}
}
