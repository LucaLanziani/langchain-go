package provider

import (
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"
)

// TestProperty4_OptionApplicationOrderIndependence tests that common options
// can be applied in any order with the same result.
//
// Property 4: Option Application Order Independence
// ∀ opts1, opts2 ∈ Permutations(commonOptions):
//
//	model1, _, _ := NewProvider(ctx, provider, opts1...)
//	model2, _, _ := NewProvider(ctx, provider, opts2...)
//	⟹ model1.config ≡ model2.config
//
// Validates: Requirement 2.6
func TestProperty4_OptionApplicationOrderIndependence(t *testing.T) {
	// Test with a representative set of common options
	testCases := []struct {
		name    string
		options []ProviderOption
	}{
		{
			name: "all common options",
			options: []ProviderOption{
				WithModel("test-model"),
				WithTemperature(0.7),
				WithMaxTokens(1000),
				WithTopP(0.9),
				WithStop([]string{"stop1", "stop2"}),
				WithAPIKey("test-key"),
				WithBaseURL("https://test.example.com"),
			},
		},
		{
			name: "subset of options",
			options: []ProviderOption{
				WithModel("gpt-4"),
				WithTemperature(0.5),
				WithMaxTokens(2000),
			},
		},
		{
			name: "minimal options",
			options: []ProviderOption{
				WithModel("claude-3"),
				WithMaxTokens(4096),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Generate all permutations of the options
			permutations := generatePermutations(tc.options)

			// Apply each permutation and collect the resulting configs
			var configs []*ProviderConfig
			for _, perm := range permutations {
				config := defaultConfig()
				for _, opt := range perm {
					opt(config)
				}
				configs = append(configs, config)
			}

			// Verify all configs are equivalent
			if len(configs) < 2 {
				t.Skip("Need at least 2 permutations to test")
			}

			baseConfig := configs[0]
			for i, config := range configs[1:] {
				if !configsEqual(baseConfig, config) {
					t.Errorf("Permutation %d produced different config:\nExpected: %+v\nGot: %+v",
						i+1, baseConfig, config)
				}
			}
		})
	}
}

// TestProperty4_QuickCheck uses property-based testing to verify option order independence
// with randomly generated configurations
func TestProperty4_QuickCheck(t *testing.T) {
	property := func(seed int64) bool {
		// Use seed for reproducible randomness
		rng := rand.New(rand.NewSource(seed))

		// Generate random configuration values
		model := randomString(rng, 10)
		temp := rng.Float64() * 2.0 // 0.0 to 2.0
		maxTokens := rng.Intn(4000) + 100
		topP := rng.Float64() // 0.0 to 1.0
		apiKey := randomString(rng, 20)
		baseURL := "https://" + randomString(rng, 15) + ".com"

		// Create options
		options := []ProviderOption{
			WithModel(model),
			WithTemperature(temp),
			WithMaxTokens(maxTokens),
			WithTopP(topP),
			WithAPIKey(apiKey),
			WithBaseURL(baseURL),
		}

		// Test two different orderings
		config1 := defaultConfig()
		for _, opt := range options {
			opt(config1)
		}

		// Shuffle options for second config
		shuffled := make([]ProviderOption, len(options))
		copy(shuffled, options)
		rng.Shuffle(len(shuffled), func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})

		config2 := defaultConfig()
		for _, opt := range shuffled {
			opt(config2)
		}

		// Configs should be equal regardless of option order
		return configsEqual(config1, config2)
	}

	if err := quick.Check(property, &quick.Config{MaxCount: 100}); err != nil {
		t.Errorf("Property violated: %v", err)
	}
}

// TestProperty4_ProviderSpecificOptionsOrderIndependence tests that provider-specific
// options also maintain order independence
func TestProperty4_ProviderSpecificOptionsOrderIndependence(t *testing.T) {
	options := []ProviderOption{
		WithModel("test-model"),
		WithProviderSpecific("key1", "value1"),
		WithProviderSpecific("key2", 42),
		WithProviderSpecific("key3", true),
		WithTemperature(0.8),
		WithProviderSpecific("key4", []string{"a", "b", "c"}),
	}

	// Generate multiple permutations
	permutations := generatePermutations(options)

	var configs []*ProviderConfig
	for _, perm := range permutations {
		config := defaultConfig()
		for _, opt := range perm {
			opt(config)
		}
		configs = append(configs, config)
	}

	// All configs should be equal
	if len(configs) < 2 {
		t.Skip("Need at least 2 permutations")
	}

	baseConfig := configs[0]
	for i, config := range configs[1:] {
		if !configsEqual(baseConfig, config) {
			t.Errorf("Permutation %d produced different config:\nExpected: %+v\nGot: %+v",
				i+1, baseConfig, config)
		}
	}
}

// TestProperty4_DuplicateOptionsLastWins tests that when the same option is applied
// multiple times, the last value wins (consistent behavior)
func TestProperty4_DuplicateOptionsLastWins(t *testing.T) {
	testCases := []struct {
		name     string
		options  []ProviderOption
		expected *ProviderConfig
	}{
		{
			name: "duplicate model",
			options: []ProviderOption{
				WithModel("model1"),
				WithModel("model2"),
				WithModel("model3"),
			},
			expected: &ProviderConfig{
				Model:            "model3",
				ProviderSpecific: make(map[string]any),
			},
		},
		{
			name: "duplicate temperature",
			options: []ProviderOption{
				WithTemperature(0.5),
				WithTemperature(0.7),
				WithTemperature(0.9),
			},
			expected: func() *ProviderConfig {
				temp := 0.9
				return &ProviderConfig{
					Temperature:      &temp,
					ProviderSpecific: make(map[string]any),
				}
			}(),
		},
		{
			name: "mixed duplicates",
			options: []ProviderOption{
				WithModel("model1"),
				WithTemperature(0.5),
				WithModel("model2"),
				WithMaxTokens(1000),
				WithTemperature(0.7),
			},
			expected: func() *ProviderConfig {
				temp := 0.7
				maxTokens := 1000
				return &ProviderConfig{
					Model:            "model2",
					Temperature:      &temp,
					MaxTokens:        &maxTokens,
					ProviderSpecific: make(map[string]any),
				}
			}(),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := defaultConfig()
			for _, opt := range tc.options {
				opt(config)
			}

			if !configsEqual(config, tc.expected) {
				t.Errorf("Config mismatch:\nExpected: %+v\nGot: %+v", tc.expected, config)
			}
		})
	}
}

// Helper functions

// configsEqual compares two ProviderConfig structs for equality
func configsEqual(c1, c2 *ProviderConfig) bool {
	if c1.Model != c2.Model {
		return false
	}

	if !floatPtrEqual(c1.Temperature, c2.Temperature) {
		return false
	}

	if !intPtrEqual(c1.MaxTokens, c2.MaxTokens) {
		return false
	}

	if !floatPtrEqual(c1.TopP, c2.TopP) {
		return false
	}

	if !stringSliceEqual(c1.Stop, c2.Stop) {
		return false
	}

	if c1.APIKey != c2.APIKey {
		return false
	}

	if c1.BaseURL != c2.BaseURL {
		return false
	}

	if !reflect.DeepEqual(c1.ProviderSpecific, c2.ProviderSpecific) {
		return false
	}

	return true
}

// floatPtrEqual compares two float64 pointers
func floatPtrEqual(a, b *float64) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return *a == *b
}

// intPtrEqual compares two int pointers
func intPtrEqual(a, b *int) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return *a == *b
}

// stringSliceEqual compares two string slices
func stringSliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// generatePermutations generates all permutations of the given options
// For large sets, this limits to a reasonable number of permutations
func generatePermutations(options []ProviderOption) [][]ProviderOption {
	if len(options) == 0 {
		return [][]ProviderOption{{}}
	}

	// Limit permutations for large sets to avoid exponential explosion
	if len(options) > 7 {
		// For large sets, just test a few random permutations
		return generateRandomPermutations(options, 20)
	}

	var result [][]ProviderOption
	permute(options, 0, &result)
	return result
}

// permute generates all permutations recursively
func permute(options []ProviderOption, start int, result *[][]ProviderOption) {
	if start == len(options)-1 {
		perm := make([]ProviderOption, len(options))
		copy(perm, options)
		*result = append(*result, perm)
		return
	}

	for i := start; i < len(options); i++ {
		options[start], options[i] = options[i], options[start]
		permute(options, start+1, result)
		options[start], options[i] = options[i], options[start]
	}
}

// generateRandomPermutations generates n random permutations of the options
func generateRandomPermutations(options []ProviderOption, n int) [][]ProviderOption {
	result := make([][]ProviderOption, n)
	rng := rand.New(rand.NewSource(42)) // Fixed seed for reproducibility

	for i := 0; i < n; i++ {
		perm := make([]ProviderOption, len(options))
		copy(perm, options)
		rng.Shuffle(len(perm), func(i, j int) {
			perm[i], perm[j] = perm[j], perm[i]
		})
		result[i] = perm
	}

	return result
}

// randomString generates a random string of given length
func randomString(rng *rand.Rand, length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[rng.Intn(len(charset))]
	}
	return string(b)
}
