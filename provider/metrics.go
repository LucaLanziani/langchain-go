package provider

import (
	"time"
)

// ProviderStats contains computed statistics for a single provider
type ProviderStats struct {
	ProviderName   string        // Name of the provider
	RequestCount   int64         // Total number of requests
	ErrorCount     int64         // Total number of errors
	SuccessCount   int64         // Total number of successful requests
	TotalLatency   time.Duration // Cumulative latency
	AverageLatency time.Duration // Average latency per request
	ErrorRate      float64       // Error rate (0.0 to 1.0)
	SuccessRate    float64       // Success rate (0.0 to 1.0)
	LastUsed       time.Time     // Last usage timestamp
}

// GetStats returns computed statistics for a specific provider.
// Returns nil if the provider doesn't exist in the metrics.
func (m *RouterMetrics) GetStats(providerName string) *ProviderStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	requestCount, exists := m.RequestCount[providerName]
	if !exists {
		return nil
	}

	errorCount := m.ErrorCount[providerName]
	totalLatency := m.TotalLatency[providerName]
	lastUsed := m.LastUsed[providerName]

	successCount := requestCount - errorCount

	var avgLatency time.Duration
	if requestCount > 0 {
		avgLatency = totalLatency / time.Duration(requestCount)
	}

	var errorRate, successRate float64
	if requestCount > 0 {
		errorRate = float64(errorCount) / float64(requestCount)
		successRate = float64(successCount) / float64(requestCount)
	}

	return &ProviderStats{
		ProviderName:   providerName,
		RequestCount:   requestCount,
		ErrorCount:     errorCount,
		SuccessCount:   successCount,
		TotalLatency:   totalLatency,
		AverageLatency: avgLatency,
		ErrorRate:      errorRate,
		SuccessRate:    successRate,
		LastUsed:       lastUsed,
	}
}

// GetAllStats returns computed statistics for all providers.
// Returns an empty map if no providers have been tracked.
func (m *RouterMetrics) GetAllStats() map[string]*ProviderStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]*ProviderStats, len(m.RequestCount))

	for providerName := range m.RequestCount {
		requestCount := m.RequestCount[providerName]
		errorCount := m.ErrorCount[providerName]
		totalLatency := m.TotalLatency[providerName]
		lastUsed := m.LastUsed[providerName]

		successCount := requestCount - errorCount

		var avgLatency time.Duration
		if requestCount > 0 {
			avgLatency = totalLatency / time.Duration(requestCount)
		}

		var errorRate, successRate float64
		if requestCount > 0 {
			errorRate = float64(errorCount) / float64(requestCount)
			successRate = float64(successCount) / float64(requestCount)
		}

		result[providerName] = &ProviderStats{
			ProviderName:   providerName,
			RequestCount:   requestCount,
			ErrorCount:     errorCount,
			SuccessCount:   successCount,
			TotalLatency:   totalLatency,
			AverageLatency: avgLatency,
			ErrorRate:      errorRate,
			SuccessRate:    successRate,
			LastUsed:       lastUsed,
		}
	}

	return result
}

// Reset clears all metrics for a specific provider.
// This is useful for resetting statistics without recreating the router.
func (m *RouterMetrics) Reset(providerName string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.RequestCount, providerName)
	delete(m.ErrorCount, providerName)
	delete(m.TotalLatency, providerName)
	delete(m.LastUsed, providerName)
}

// ResetAll clears all metrics for all providers.
func (m *RouterMetrics) ResetAll() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.RequestCount = make(map[string]int64)
	m.ErrorCount = make(map[string]int64)
	m.TotalLatency = make(map[string]time.Duration)
	m.LastUsed = make(map[string]time.Time)
}
