package provider

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
)

// SelectProvider uses an LLM to analyze the request and select the most appropriate provider.
// It caches routing decisions for similar requests to minimize LLM calls.
func (s *LLMRoutingStrategy) SelectProvider(ctx context.Context, reqCtx RequestContext, providers map[string]llms.ChatModel) (string, error) {
	if len(providers) == 0 {
		return "", ErrNoProvidersAvailable
	}

	if s.model == nil {
		// Fallback to first available provider if no LLM configured
		return anyKey(providers), fmt.Errorf("LLM routing model not configured, using fallback")
	}

	// Step 1: Generate cache key from request characteristics
	cacheKey := s.generateCacheKey(reqCtx)

	// Step 2: Check cache for recent routing decision
	s.mu.RLock()
	if entry, found := s.cache[cacheKey]; found {
		// Check if cache entry is still valid
		if time.Now().Before(entry.expiresAt) {
			// Validate cached provider still exists
			if _, exists := providers[entry.providerName]; exists {
				s.mu.RUnlock()
				return entry.providerName, nil
			}
		}
	}
	s.mu.RUnlock()

	// Step 3: Build prompt for LLM routing decision
	systemPrompt := s.systemPrompt
	if systemPrompt == "" {
		systemPrompt = s.buildDefaultSystemPrompt()
	}

	userPrompt := s.buildRoutingPrompt(reqCtx, providers)

	messages := []core.Message{
		core.NewSystemMessage(systemPrompt),
		core.NewHumanMessage(userPrompt),
	}

	// Step 4: Call LLM to get routing decision
	response, err := s.model.Invoke(ctx, messages)
	if err != nil {
		// Fallback to first available provider if LLM fails
		return anyKey(providers), fmt.Errorf("LLM routing failed: %w", err)
	}

	// Step 5: Parse LLM response to extract provider name
	providerName := s.parseProviderName(response.Content)

	// Step 6: Validate provider exists
	if _, exists := providers[providerName]; !exists {
		// LLM returned invalid provider, use first available
		return anyKey(providers), fmt.Errorf("%w: LLM returned invalid provider %s", ErrInvalidProviderFromLLM, providerName)
	}

	// Step 7: Cache the routing decision
	s.mu.Lock()
	if s.cache == nil {
		s.cache = make(map[string]*llmCacheEntry)
	}

	// Cleanup expired entries periodically to prevent unbounded growth
	// Only cleanup if cache is getting large (>100 entries)
	if len(s.cache) > 100 {
		now := time.Now()
		for key, entry := range s.cache {
			if now.After(entry.expiresAt) {
				delete(s.cache, key)
			}
		}
	}

	s.cache[cacheKey] = &llmCacheEntry{
		providerName: providerName,
		expiresAt:    time.Now().Add(s.cacheTTL),
	}
	s.mu.Unlock()

	return providerName, nil
}

// OnSuccess is a no-op for LLMRoutingStrategy.
// The LLM makes routing decisions based on request characteristics, not historical performance.
func (s *LLMRoutingStrategy) OnSuccess(ctx context.Context, providerName string, latency time.Duration) {
	// No-op: LLMRoutingStrategy doesn't adapt based on feedback
	// Future enhancement: could incorporate performance metrics into routing prompt
}

// OnError is a no-op for LLMRoutingStrategy.
// The LLM makes routing decisions based on request characteristics, not historical performance.
func (s *LLMRoutingStrategy) OnError(ctx context.Context, providerName string, err error) {
	// No-op: LLMRoutingStrategy doesn't adapt based on feedback
	// Future enhancement: could incorporate error patterns into routing prompt
}

// generateCacheKey creates a deterministic cache key from request characteristics.
// Similar requests should produce the same key to maximize cache hits.
// Optimized to minimize allocations.
func (s *LLMRoutingStrategy) generateCacheKey(reqCtx RequestContext) string {
	// Pre-allocate buffer for key parts (avoid multiple allocations)
	var sb strings.Builder
	sb.Grow(64) // Pre-allocate reasonable size

	sb.WriteString("c:")
	sb.WriteString(reqCtx.Complexity)
	sb.WriteByte('|')
	sb.WriteString("p:")
	sb.WriteString(reqCtx.Priority)
	sb.WriteByte('|')
	sb.WriteString("t:")
	if reqCtx.HasToolCalls {
		sb.WriteByte('1')
	} else {
		sb.WriteByte('0')
	}
	sb.WriteByte('|')
	sb.WriteString("tk:")

	// Bucket token count to avoid cache misses for similar sizes
	tokenBucket := reqCtx.TotalTokens / 1000 // Bucket by 1000 tokens
	sb.WriteString(fmt.Sprintf("%d", tokenBucket))

	// Hash for consistent key length
	keyString := sb.String()
	hash := sha256.Sum256([]byte(keyString))
	return hex.EncodeToString(hash[:16]) // Use first 16 bytes for shorter keys
}

// buildDefaultSystemPrompt creates the default system prompt for routing decisions.
func (s *LLMRoutingStrategy) buildDefaultSystemPrompt() string {
	return `You are a routing assistant that selects the best LLM provider for each request.
Analyze the request characteristics and available providers to make an optimal routing decision.
Consider factors like complexity, token count, tool usage, and provider strengths.
Respond with ONLY the provider name, nothing else.`
}

// buildRoutingPrompt creates the user prompt with request characteristics and provider descriptions.
func (s *LLMRoutingStrategy) buildRoutingPrompt(reqCtx RequestContext, providers map[string]llms.ChatModel) string {
	var prompt strings.Builder

	prompt.WriteString("Request Characteristics:\n")
	prompt.WriteString(fmt.Sprintf("- Message Count: %d\n", reqCtx.MessageCount))
	prompt.WriteString(fmt.Sprintf("- Estimated Tokens: %d\n", reqCtx.TotalTokens))
	prompt.WriteString(fmt.Sprintf("- Has Tool Calls: %t\n", reqCtx.HasToolCalls))
	prompt.WriteString(fmt.Sprintf("- Complexity: %s\n", reqCtx.Complexity))
	prompt.WriteString(fmt.Sprintf("- Priority: %s\n\n", reqCtx.Priority))

	prompt.WriteString("Available Providers:\n")

	// Get provider names in deterministic order
	providerNames := make([]string, 0, len(providers))
	for name := range providers {
		providerNames = append(providerNames, name)
	}
	sort.Strings(providerNames)

	for _, name := range providerNames {
		description := s.providerDescriptions[name]
		if description == "" {
			description = "General purpose LLM provider"
		}
		prompt.WriteString(fmt.Sprintf("- %s: %s\n", name, description))
	}

	prompt.WriteString("\nRespond with ONLY the provider name (e.g., 'openai' or 'anthropic'). No explanation.")

	return prompt.String()
}

// parseProviderName extracts the provider name from the LLM response.
// It handles various response formats and cleans up the output.
func (s *LLMRoutingStrategy) parseProviderName(response string) string {
	// Trim whitespace and convert to lowercase
	providerName := strings.TrimSpace(response)
	providerName = strings.ToLower(providerName)

	// Remove common prefixes/suffixes
	providerName = strings.TrimPrefix(providerName, "provider:")
	providerName = strings.TrimPrefix(providerName, "provider ")
	providerName = strings.TrimSuffix(providerName, ".")
	providerName = strings.TrimSpace(providerName)

	// Remove quotes if present
	providerName = strings.Trim(providerName, "\"'`")

	return providerName
}

// anyKey returns any key from the providers map.
// Used as a fallback when LLM routing fails.
func anyKey(providers map[string]llms.ChatModel) string {
	for name := range providers {
		return name
	}
	return ""
}
