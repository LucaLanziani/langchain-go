package provider

import (
	"errors"
	"fmt"
)

// Sentinel errors for provider operations
var (
	// ErrUnknownProvider is returned when an invalid provider type is specified
	ErrUnknownProvider = errors.New("unknown provider type")

	// ErrInvalidConfig is returned when provider configuration is invalid
	ErrInvalidConfig = errors.New("invalid provider configuration")

	// ErrDuplicateProviderName is returned when router entries have duplicate names
	ErrDuplicateProviderName = errors.New("duplicate provider name in router entries")

	// ErrEmptyProviderList is returned when router is created with no providers
	ErrEmptyProviderList = errors.New("router requires at least one provider")

	// ErrProviderNotFound is returned when a requested provider doesn't exist
	ErrProviderNotFound = errors.New("provider not found")

	// ErrNoFallbackAvailable is returned when fallback is needed but no providers are available
	ErrNoFallbackAvailable = errors.New("no fallback provider available")

	// ErrMaxRetriesExceeded is returned when all retry attempts have been exhausted
	ErrMaxRetriesExceeded = errors.New("maximum retry attempts exceeded")

	// ErrInvalidProviderFromLLM is returned when LLM routing returns an invalid provider name
	ErrInvalidProviderFromLLM = errors.New("LLM returned invalid provider name")

	// ErrMissingRequiredField is returned when a required configuration field is missing
	ErrMissingRequiredField = errors.New("missing required configuration field")

	// ErrInvalidFieldValue is returned when a configuration field has an invalid value
	ErrInvalidFieldValue = errors.New("invalid configuration field value")

	// ErrAuthenticationFailed is returned when provider authentication fails
	ErrAuthenticationFailed = errors.New("provider authentication failed")

	// ErrProviderInitialization is returned when provider initialization fails
	ErrProviderInitialization = errors.New("provider initialization failed")

	// ErrRouterClosed is returned when operations are attempted on a closed router
	ErrRouterClosed = errors.New("router has been closed")

	// ErrNoProvidersAvailable is returned when no providers are available for routing
	ErrNoProvidersAvailable = errors.New("no providers available")
)

// ValidationError represents a configuration validation error
type ValidationError struct {
	Field   string
	Value   any
	Message string
}

func (e *ValidationError) Error() string {
	// Sanitize sensitive field values
	sanitizedValue := sanitizeValue(e.Field, e.Value)
	return fmt.Sprintf("validation error for field '%s': %s (value: %v)", e.Field, e.Message, sanitizedValue)
}

// ProviderError wraps errors with provider context
type ProviderError struct {
	ProviderType ProviderType
	ProviderName string
	Operation    string
	Err          error
	// Additional context for debugging (never includes sensitive data)
	Context map[string]string
}

func (e *ProviderError) Error() string {
	var msg string
	if e.ProviderName != "" {
		msg = fmt.Sprintf("provider error [%s/%s] during %s: %v", e.ProviderType, e.ProviderName, e.Operation, e.Err)
	} else {
		msg = fmt.Sprintf("provider error [%s] during %s: %v", e.ProviderType, e.Operation, e.Err)
	}

	// Add context if available
	if len(e.Context) > 0 {
		msg += " (context:"
		first := true
		for k, v := range e.Context {
			if !first {
				msg += ","
			}
			msg += fmt.Sprintf(" %s=%s", k, v)
			first = false
		}
		msg += ")"
	}

	return msg
}

func (e *ProviderError) Unwrap() error {
	return e.Err
}

// RoutingError wraps errors that occur during request routing
type RoutingError struct {
	Strategy           string
	AvailableProviders []string
	RequestComplexity  string
	Err                error
}

func (e *RoutingError) Error() string {
	msg := fmt.Sprintf("routing error [strategy: %s]: %v", e.Strategy, e.Err)
	if len(e.AvailableProviders) > 0 {
		msg += fmt.Sprintf(" (available providers: %v)", e.AvailableProviders)
	}
	if e.RequestComplexity != "" {
		msg += fmt.Sprintf(" (request complexity: %s)", e.RequestComplexity)
	}
	return msg
}

func (e *RoutingError) Unwrap() error {
	return e.Err
}

// FallbackError wraps errors that occur during fallback attempts
type FallbackError struct {
	FailedProvider     string
	AttemptedFallbacks []string
	FallbackStrategy   string
	Err                error
}

func (e *FallbackError) Error() string {
	msg := fmt.Sprintf("fallback error after %s failed", e.FailedProvider)
	if e.FallbackStrategy != "" {
		msg += fmt.Sprintf(" [strategy: %s]", e.FallbackStrategy)
	}
	if len(e.AttemptedFallbacks) > 0 {
		msg += fmt.Sprintf(" (attempted: %v)", e.AttemptedFallbacks)
	}
	msg += fmt.Sprintf(": %v", e.Err)
	return msg
}

func (e *FallbackError) Unwrap() error {
	return e.Err
}

// NewValidationError creates a new validation error
func NewValidationError(field string, value any, message string) error {
	return &ValidationError{
		Field:   field,
		Value:   value,
		Message: message,
	}
}

// NewProviderError creates a new provider error with context
func NewProviderError(providerType ProviderType, providerName, operation string, err error) error {
	return &ProviderError{
		ProviderType: providerType,
		ProviderName: providerName,
		Operation:    operation,
		Err:          err,
		Context:      make(map[string]string),
	}
}

// NewProviderErrorWithContext creates a new provider error with additional context
func NewProviderErrorWithContext(providerType ProviderType, providerName, operation string, err error, context map[string]string) error {
	// Sanitize context to ensure no sensitive information
	sanitizedContext := make(map[string]string)
	for k, v := range context {
		sanitizedContext[k] = sanitizeContextValue(k, v)
	}

	return &ProviderError{
		ProviderType: providerType,
		ProviderName: providerName,
		Operation:    operation,
		Err:          err,
		Context:      sanitizedContext,
	}
}

// NewRoutingError creates a new routing error
func NewRoutingError(strategy string, err error) error {
	return &RoutingError{
		Strategy: strategy,
		Err:      err,
	}
}

// NewRoutingErrorWithContext creates a new routing error with additional context
func NewRoutingErrorWithContext(strategy string, availableProviders []string, requestComplexity string, err error) error {
	return &RoutingError{
		Strategy:           strategy,
		AvailableProviders: availableProviders,
		RequestComplexity:  requestComplexity,
		Err:                err,
	}
}

// NewFallbackError creates a new fallback error
func NewFallbackError(failedProvider string, attemptedFallbacks []string, err error) error {
	return &FallbackError{
		FailedProvider:     failedProvider,
		AttemptedFallbacks: attemptedFallbacks,
		Err:                err,
	}
}

// NewFallbackErrorWithStrategy creates a new fallback error with strategy information
func NewFallbackErrorWithStrategy(failedProvider string, attemptedFallbacks []string, fallbackStrategy string, err error) error {
	return &FallbackError{
		FailedProvider:     failedProvider,
		AttemptedFallbacks: attemptedFallbacks,
		FallbackStrategy:   fallbackStrategy,
		Err:                err,
	}
}

// sanitizeValue sanitizes field values to prevent exposure of sensitive information
func sanitizeValue(fieldName string, value any) any {
	if value == nil {
		return nil
	}

	// List of sensitive field names (case-insensitive check)
	sensitiveFields := []string{"apikey", "api_key", "token", "password", "secret", "credential", "auth"}

	fieldLower := ""
	if fieldName != "" {
		fieldLower = fmt.Sprintf("%v", fieldName)
		// Simple lowercase conversion for ASCII
		fieldLower = toLower(fieldLower)
	}

	// Check if field name contains sensitive keywords
	for _, sensitive := range sensitiveFields {
		if contains(fieldLower, sensitive) {
			return "[REDACTED]"
		}
	}

	return value
}

// sanitizeContextValue sanitizes context values to prevent exposure of sensitive information
func sanitizeContextValue(key, value string) string {
	// List of sensitive context keys
	sensitiveKeys := []string{"apikey", "api_key", "token", "password", "secret", "credential", "auth", "authorization"}

	keyLower := toLower(key)

	// Check if key contains sensitive keywords
	for _, sensitive := range sensitiveKeys {
		if contains(keyLower, sensitive) {
			return "[REDACTED]"
		}
	}

	return value
}

// toLower converts ASCII string to lowercase (simple implementation)
func toLower(s string) string {
	result := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			result[i] = c + ('a' - 'A')
		} else {
			result[i] = c
		}
	}
	return string(result)
}

// contains checks if string s contains substring substr (case-sensitive)
func contains(s, substr string) bool {
	return len(s) >= len(substr) && indexOfSubstring(s, substr) >= 0
}

// indexOfSubstring returns the index of substr in s, or -1 if not found
func indexOfSubstring(s, substr string) int {
	if len(substr) == 0 {
		return 0
	}
	if len(substr) > len(s) {
		return -1
	}

	for i := 0; i <= len(s)-len(substr); i++ {
		match := true
		for j := 0; j < len(substr); j++ {
			if s[i+j] != substr[j] {
				match = false
				break
			}
		}
		if match {
			return i
		}
	}
	return -1
}
