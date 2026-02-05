package core

import (
	"github.com/google/uuid"
)

// RunnableConfig holds configuration for a Runnable invocation.
// All fields are optional and can be set via Option functions.
type RunnableConfig struct {
	// Tags for this call and any sub-calls (used for filtering and tracing).
	Tags []string

	// Metadata for this call and any sub-calls.
	Metadata map[string]any

	// Callbacks are the callback handlers for this call.
	Callbacks []CallbackHandler

	// RunName overrides the default name for tracing.
	RunName string

	// MaxConcurrency limits parallel calls in Batch operations.
	// 0 means no limit.
	MaxConcurrency int

	// RecursionLimit is the maximum recursion depth. Default is 25.
	RecursionLimit int

	// Configurable holds runtime values for configurable fields.
	Configurable map[string]any

	// RunID is a unique identifier for this run. Auto-generated if empty.
	RunID string

	// Stop sequences to pass to the model.
	Stop []string
}

// DefaultConfig returns a RunnableConfig with sensible defaults.
func DefaultConfig() *RunnableConfig {
	return &RunnableConfig{
		RecursionLimit: 25,
		RunID:          uuid.New().String(),
		Metadata:       make(map[string]any),
	}
}

// Option is a function that modifies a RunnableConfig.
type Option func(*RunnableConfig)

// ApplyOptions applies a set of options to a config, starting from defaults.
func ApplyOptions(opts ...Option) *RunnableConfig {
	cfg := DefaultConfig()
	for _, opt := range opts {
		opt(cfg)
	}
	return cfg
}

// MergeOptions merges a base config with additional options.
func MergeOptions(base *RunnableConfig, opts ...Option) *RunnableConfig {
	if base == nil {
		base = DefaultConfig()
	}
	for _, opt := range opts {
		opt(base)
	}
	return base
}

// WithTags adds tags to the config.
func WithTags(tags ...string) Option {
	return func(c *RunnableConfig) {
		c.Tags = append(c.Tags, tags...)
	}
}

// WithMetadata adds metadata to the config.
func WithMetadata(metadata map[string]any) Option {
	return func(c *RunnableConfig) {
		if c.Metadata == nil {
			c.Metadata = make(map[string]any)
		}
		for k, v := range metadata {
			c.Metadata[k] = v
		}
	}
}

// WithCallbacks sets the callback handlers.
func WithCallbacks(handlers ...CallbackHandler) Option {
	return func(c *RunnableConfig) {
		c.Callbacks = append(c.Callbacks, handlers...)
	}
}

// WithRunName sets the run name for tracing.
func WithRunName(name string) Option {
	return func(c *RunnableConfig) {
		c.RunName = name
	}
}

// WithMaxConcurrency sets the maximum number of parallel calls in Batch.
func WithMaxConcurrency(n int) Option {
	return func(c *RunnableConfig) {
		c.MaxConcurrency = n
	}
}

// WithRecursionLimit sets the recursion limit.
func WithRecursionLimit(n int) Option {
	return func(c *RunnableConfig) {
		c.RecursionLimit = n
	}
}

// WithRunID sets a specific run ID.
func WithRunID(id string) Option {
	return func(c *RunnableConfig) {
		c.RunID = id
	}
}

// WithStop sets stop sequences.
func WithStop(stop ...string) Option {
	return func(c *RunnableConfig) {
		c.Stop = stop
	}
}

// WithConfigurable sets configurable runtime values.
func WithConfigurable(values map[string]any) Option {
	return func(c *RunnableConfig) {
		if c.Configurable == nil {
			c.Configurable = make(map[string]any)
		}
		for k, v := range values {
			c.Configurable[k] = v
		}
	}
}
