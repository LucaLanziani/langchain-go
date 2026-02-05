// Package memory provides memory implementations for maintaining conversation
// state across interactions.
package memory

import (
	"context"
)

// Memory is the interface for conversation memory.
// Memory loads relevant context before a chain runs and saves context after.
type Memory interface {
	// MemoryVariables returns the keys this memory will add to chain inputs.
	MemoryVariables() []string

	// LoadMemoryVariables returns key-value pairs to be added to chain inputs.
	LoadMemoryVariables(ctx context.Context, inputs map[string]any) (map[string]any, error)

	// SaveContext saves the input and output of a chain run to memory.
	SaveContext(ctx context.Context, inputs map[string]any, outputs map[string]any) error

	// Clear resets the memory.
	Clear(ctx context.Context) error
}
