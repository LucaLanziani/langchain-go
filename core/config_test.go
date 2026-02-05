package core

import (
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.RecursionLimit != 25 {
		t.Errorf("expected RecursionLimit 25, got %d", cfg.RecursionLimit)
	}
	if cfg.RunID == "" {
		t.Error("expected non-empty RunID")
	}
	if cfg.Metadata == nil {
		t.Error("expected non-nil Metadata")
	}
}

func TestApplyOptions(t *testing.T) {
	cfg := ApplyOptions(
		WithTags("tag1", "tag2"),
		WithRunName("test-run"),
		WithMaxConcurrency(5),
		WithRecursionLimit(10),
		WithStop("stop1"),
		WithMetadata(map[string]any{"key": "value"}),
	)

	if len(cfg.Tags) != 2 || cfg.Tags[0] != "tag1" {
		t.Errorf("expected tags [tag1, tag2], got %v", cfg.Tags)
	}
	if cfg.RunName != "test-run" {
		t.Errorf("expected RunName 'test-run', got %q", cfg.RunName)
	}
	if cfg.MaxConcurrency != 5 {
		t.Errorf("expected MaxConcurrency 5, got %d", cfg.MaxConcurrency)
	}
	if cfg.RecursionLimit != 10 {
		t.Errorf("expected RecursionLimit 10, got %d", cfg.RecursionLimit)
	}
	if len(cfg.Stop) != 1 || cfg.Stop[0] != "stop1" {
		t.Errorf("expected stop [stop1], got %v", cfg.Stop)
	}
	if cfg.Metadata["key"] != "value" {
		t.Errorf("expected metadata key=value, got %v", cfg.Metadata)
	}
}

func TestWithConfigurable(t *testing.T) {
	cfg := ApplyOptions(
		WithConfigurable(map[string]any{"model": "gpt-4"}),
	)
	if cfg.Configurable["model"] != "gpt-4" {
		t.Errorf("expected configurable model=gpt-4, got %v", cfg.Configurable)
	}
}

func TestMergeOptions(t *testing.T) {
	base := &RunnableConfig{
		Tags:           []string{"base-tag"},
		RecursionLimit: 25,
	}
	merged := MergeOptions(base, WithTags("new-tag"), WithMaxConcurrency(3))

	if len(merged.Tags) != 2 {
		t.Errorf("expected 2 tags, got %d", len(merged.Tags))
	}
	if merged.MaxConcurrency != 3 {
		t.Errorf("expected MaxConcurrency 3, got %d", merged.MaxConcurrency)
	}
}
