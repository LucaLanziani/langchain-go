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
)

// ValidationError represents a configuration validation error
type ValidationError struct {
	Field   string
	Value   any
	Message string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("validation error for field '%s': %s (value: %v)", e.Field, e.Message, e.Value)
}

// ProviderError wraps errors with provider context
type ProviderError struct {
	ProviderType ProviderType
	ProviderName string
	Operation    string
	Err          error
}

func (e *ProviderError) Error() string {
	if e.ProviderName != "" {
		return fmt.Sprintf("provider error [%s/%s] during %s: %v", e.ProviderType, e.ProviderName, e.Operation, e.Err)
	}
	return fmt.Sprintf("provider error [%s] during %s: %v", e.ProviderType, e.Operation, e.Err)
}

func (e *ProviderError) Unwrap() error {
	return e.Err
}

// RoutingError wraps errors that occur during request routing
type RoutingError struct {
	Strategy string
	Err      error
}

func (e *RoutingError) Error() string {
	return fmt.Sprintf("routing error [strategy: %s]: %v", e.Strategy, e.Err)
}

func (e *RoutingError) Unwrap() error {
	return e.Err
}

// FallbackError wraps errors that occur during fallback attempts
type FallbackError struct {
	FailedProvider     string
	AttemptedFallbacks []string
	Err                error
}

func (e *FallbackError) Error() string {
	return fmt.Sprintf("fallback error after %s failed (attempted: %v): %v", e.FailedProvider, e.AttemptedFallbacks, e.Err)
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
	}
}

// NewRoutingError creates a new routing error
func NewRoutingError(strategy string, err error) error {
	return &RoutingError{
		Strategy: strategy,
		Err:      err,
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
