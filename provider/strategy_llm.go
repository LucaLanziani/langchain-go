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

	pending, waitForLeader, providerName, err := s.prepareRequest(cacheKey, providers)
	if pending == nil {
		return providerName, err
	}
	if waitForLeader {
		providerName, err = s.waitForPending(ctx, pending, providers)
		if providerName == "" && err != nil {
			return anyKey(providers), err
		}
		return providerName, err
	}

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
		pending.err = fmt.Errorf("LLM routing failed: %w", err)
		s.completeRequest(cacheKey, pending, "", nil)
		return anyKey(providers), pending.err
	}

	// Step 5: Parse LLM response to extract provider name
	providerName = s.parseProviderName(response.Content)

	// Step 6: Validate provider exists
	if _, exists := providers[providerName]; !exists {
		// LLM returned invalid provider, use first available
		pending.err = fmt.Errorf("%w: LLM returned invalid provider %s", ErrInvalidProviderFromLLM, providerName)
		s.completeRequest(cacheKey, pending, "", nil)
		return anyKey(providers), pending.err
	}

	// Step 7: Cache the routing decision
	s.completeRequest(cacheKey, pending, providerName, &llmCacheEntry{
		providerName: providerName,
		expiresAt:    time.Now().Add(s.cacheTTL),
	})

	return providerName, nil
}

func (s *LLMRoutingStrategy) prepareRequest(cacheKey string, providers map[string]llms.ChatModel) (*llmPendingCall, bool, string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if entry, found := s.cache[cacheKey]; found {
		if time.Now().Before(entry.expiresAt) {
			if _, exists := providers[entry.providerName]; exists {
				return nil, false, entry.providerName, nil
			}
		}
		delete(s.cache, cacheKey)
	}

	if s.inFlight == nil {
		s.inFlight = make(map[string]*llmPendingCall)
	}
	if pending, found := s.inFlight[cacheKey]; found {
		return pending, true, "", nil
	}

	pending := &llmPendingCall{done: make(chan struct{})}
	s.inFlight[cacheKey] = pending
	return pending, false, "", nil
}

func (s *LLMRoutingStrategy) waitForPending(ctx context.Context, pending *llmPendingCall, providers map[string]llms.ChatModel) (string, error) {
	select {
	case <-pending.done:
		if pending.providerName == "" {
			return "", pending.err
		}
		if _, exists := providers[pending.providerName]; !exists {
			return anyKey(providers), fmt.Errorf("%w: provider %s no longer available", ErrInvalidProviderFromLLM, pending.providerName)
		}
		return pending.providerName, pending.err
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func (s *LLMRoutingStrategy) completeRequest(cacheKey string, pending *llmPendingCall, providerName string, entry *llmCacheEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if entry != nil {
		if s.cache == nil {
			s.cache = make(map[string]*llmCacheEntry)
		}
		if len(s.cache) > 100 {
			now := time.Now()
			for key, cachedEntry := range s.cache {
				if now.After(cachedEntry.expiresAt) {
					delete(s.cache, key)
				}
			}
		}
		s.cache[cacheKey] = entry
	}

	if pending != nil {
		pending.providerName = providerName
		close(pending.done)
		delete(s.inFlight, cacheKey)
	}
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
	names := make([]string, 0, len(providers))
	for name := range providers {
		names = append(names, name)
	}
	sort.Strings(names)
	if len(names) > 0 {
		return names[0]
	}
	return ""
}
