package runnable

import (
	"context"
	"fmt"

	"github.com/langchain-go/langchain-go/core"
)

// BranchCondition pairs a condition function with a runnable.
type BranchCondition[I, O any] struct {
	Condition func(input I) bool
	Runnable  core.Runnable[I, O]
}

// Branch selects which runnable to execute based on conditions.
// It evaluates conditions in order and runs the first one that returns true.
// If no conditions match, the default runnable is used.
// It implements Runnable[I, O].
type Branch[I, O any] struct {
	conditions     []BranchCondition[I, O]
	defaultBranch  core.Runnable[I, O]
	name           string
}

// NewBranch creates a new Branch runnable.
// conditions are evaluated in order; the first true condition's runnable is executed.
// defaultBranch is used when no conditions match.
func NewBranch[I, O any](
	conditions []BranchCondition[I, O],
	defaultBranch core.Runnable[I, O],
) *Branch[I, O] {
	return &Branch[I, O]{
		conditions:    conditions,
		defaultBranch: defaultBranch,
	}
}

// WithName sets the name for tracing.
func (b *Branch[I, O]) WithName(name string) *Branch[I, O] {
	b.name = name
	return b
}

// GetName returns the name.
func (b *Branch[I, O]) GetName() string {
	if b.name != "" {
		return b.name
	}
	return "RunnableBranch"
}

// Invoke evaluates conditions and runs the matching branch.
func (b *Branch[I, O]) Invoke(ctx context.Context, input I, opts ...core.Option) (O, error) {
	for _, cond := range b.conditions {
		if cond.Condition(input) {
			return cond.Runnable.Invoke(ctx, input, opts...)
		}
	}
	if b.defaultBranch != nil {
		return b.defaultBranch.Invoke(ctx, input, opts...)
	}
	var zero O
	return zero, fmt.Errorf("no branch condition matched and no default branch provided")
}

// Stream evaluates conditions and streams from the matching branch.
func (b *Branch[I, O]) Stream(ctx context.Context, input I, opts ...core.Option) (*core.StreamIterator[O], error) {
	for _, cond := range b.conditions {
		if cond.Condition(input) {
			return cond.Runnable.Stream(ctx, input, opts...)
		}
	}
	if b.defaultBranch != nil {
		return b.defaultBranch.Stream(ctx, input, opts...)
	}
	return nil, fmt.Errorf("no branch condition matched and no default branch provided")
}

// Batch runs the branch for multiple inputs.
func (b *Branch[I, O]) Batch(ctx context.Context, inputs []I, opts ...core.Option) ([]O, error) {
	results := make([]O, len(inputs))
	for i, input := range inputs {
		result, err := b.Invoke(ctx, input, opts...)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}
