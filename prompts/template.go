// Package prompts provides prompt template types for formatting inputs
// into prompts for language models.
package prompts

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/langchain-go/langchain-go/core"
)

var templateVarRegex = regexp.MustCompile(`\{(\w+)\}`)

// PromptTemplate is a string template that formats variables into a prompt.
// It implements Runnable[map[string]any, string].
type PromptTemplate struct {
	// Template is the template string with {variable} placeholders.
	Template string

	// InputVariables is the list of variable names expected.
	InputVariables []string

	// PartialVariables are pre-filled variables.
	PartialVariables map[string]any

	name string
}

// NewPromptTemplate creates a PromptTemplate from a template string.
// It automatically extracts input variables from {variable} placeholders.
func NewPromptTemplate(template string) *PromptTemplate {
	vars := extractVariables(template)
	return &PromptTemplate{
		Template:       template,
		InputVariables: vars,
	}
}

// WithName sets the name for tracing.
func (p *PromptTemplate) WithName(name string) *PromptTemplate {
	p.name = name
	return p
}

// WithPartialVariables sets partial variables.
func (p *PromptTemplate) WithPartialVariables(vars map[string]any) *PromptTemplate {
	p.PartialVariables = vars
	return p
}

// GetName returns the name of this prompt template.
func (p *PromptTemplate) GetName() string {
	if p.name != "" {
		return p.name
	}
	return "PromptTemplate"
}

// Format applies the variables to the template and returns the formatted string.
func (p *PromptTemplate) Format(values map[string]any) (string, error) {
	merged := make(map[string]any)
	for k, v := range p.PartialVariables {
		merged[k] = v
	}
	for k, v := range values {
		merged[k] = v
	}

	result := p.Template
	for _, varName := range p.InputVariables {
		val, ok := merged[varName]
		if !ok {
			if _, partialOK := p.PartialVariables[varName]; !partialOK {
				return "", fmt.Errorf("missing required variable: %s", varName)
			}
			continue
		}
		result = strings.ReplaceAll(result, "{"+varName+"}", fmt.Sprintf("%v", val))
	}
	// Also replace any partial variables not in InputVariables.
	for k, v := range p.PartialVariables {
		result = strings.ReplaceAll(result, "{"+k+"}", fmt.Sprintf("%v", v))
	}
	return result, nil
}

// Invoke formats the template with the given input map.
func (p *PromptTemplate) Invoke(ctx context.Context, input map[string]any, opts ...core.Option) (string, error) {
	return p.Format(input)
}

// Stream returns a single-chunk stream of the formatted result.
func (p *PromptTemplate) Stream(ctx context.Context, input map[string]any, opts ...core.Option) (*core.StreamIterator[string], error) {
	result, err := p.Format(input)
	if err != nil {
		return nil, err
	}
	ch := make(chan core.StreamChunk[string], 1)
	ch <- core.StreamChunk[string]{Value: result}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

// Batch formats the template with multiple input maps.
func (p *PromptTemplate) Batch(ctx context.Context, inputs []map[string]any, opts ...core.Option) ([]string, error) {
	results := make([]string, len(inputs))
	for i, input := range inputs {
		result, err := p.Format(input)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// extractVariables finds all {variable} placeholders in a template string.
func extractVariables(template string) []string {
	matches := templateVarRegex.FindAllStringSubmatch(template, -1)
	seen := make(map[string]bool)
	var vars []string
	for _, match := range matches {
		if len(match) > 1 && !seen[match[1]] {
			seen[match[1]] = true
			vars = append(vars, match[1])
		}
	}
	return vars
}
