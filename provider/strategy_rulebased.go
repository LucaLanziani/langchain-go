package provider

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/LucaLanziani/langchain-go/llms"
)

// NewRuleBasedStrategy creates a new RuleBasedStrategy.
// Rules are automatically sorted by priority (highest first).
func NewRuleBasedStrategy(rules []RoutingRule, defaultProvider string) *RuleBasedStrategy {
	// Sort rules by priority (descending)
	sortedRules := make([]RoutingRule, len(rules))
	copy(sortedRules, rules)
	sort.Slice(sortedRules, func(i, j int) bool {
		return sortedRules[i].Priority > sortedRules[j].Priority
	})

	return &RuleBasedStrategy{
		rules:           sortedRules,
		defaultProvider: defaultProvider,
	}
}

// SelectProvider evaluates rules in priority order and returns the first match.
// If no rules match, returns the default provider.
func (s *RuleBasedStrategy) SelectProvider(ctx context.Context, reqCtx RequestContext, providers map[string]llms.ChatModel) (string, error) {
	if len(providers) == 0 {
		return "", ErrNoProvidersAvailable
	}

	// Evaluate rules in priority order
	for _, rule := range s.rules {
		if rule.Condition(reqCtx) {
			// Verify provider exists
			if _, exists := providers[rule.Provider]; exists {
				return rule.Provider, nil
			}
			// Rule matched but provider doesn't exist - continue to next rule
		}
	}

	// No rules matched - use default provider
	if s.defaultProvider != "" {
		if _, exists := providers[s.defaultProvider]; exists {
			return s.defaultProvider, nil
		}
		return "", fmt.Errorf("%w: default provider %s not found", ErrProviderNotFound, s.defaultProvider)
	}

	// No default provider - return first available
	names := make([]string, 0, len(providers))
	for name := range providers {
		names = append(names, name)
	}
	sort.Strings(names)
	return names[0], nil
}

// OnSuccess is a no-op for RuleBasedStrategy.
func (s *RuleBasedStrategy) OnSuccess(ctx context.Context, providerName string, latency time.Duration) {
	// No-op: RuleBasedStrategy doesn't adapt based on feedback
}

// OnError is a no-op for RuleBasedStrategy.
func (s *RuleBasedStrategy) OnError(ctx context.Context, providerName string, err error) {
	// No-op: RuleBasedStrategy doesn't adapt based on feedback
}

// AddRule adds a new rule to the strategy.
// Rules are automatically re-sorted by priority.
func (s *RuleBasedStrategy) AddRule(rule RoutingRule) {
	s.rules = append(s.rules, rule)
	sort.Slice(s.rules, func(i, j int) bool {
		return s.rules[i].Priority > s.rules[j].Priority
	})
}

// RemoveRule removes a rule by name.
func (s *RuleBasedStrategy) RemoveRule(name string) {
	for i, rule := range s.rules {
		if rule.Name == name {
			s.rules = append(s.rules[:i], s.rules[i+1:]...)
			return
		}
	}
}

// GetRules returns a copy of all rules.
func (s *RuleBasedStrategy) GetRules() []RoutingRule {
	rules := make([]RoutingRule, len(s.rules))
	copy(rules, s.rules)
	return rules
}
