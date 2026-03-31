package loaders

import (
	"context"
	"fmt"
	"io/fs"
	"path/filepath"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/textsplitters"
)

// LoaderFactory is a function that creates a DocumentLoader for the given path.
type LoaderFactory func(path string) DocumentLoader

// defaultLoaderMapping maps file extensions to their loader factories.
var defaultLoaderMapping = map[string]LoaderFactory{
	".txt":  func(p string) DocumentLoader { return NewTextLoader(p) },
	".md":   func(p string) DocumentLoader { return NewMarkdownLoader(p) },
	".html": func(p string) DocumentLoader { return NewHTMLLoader(p) },
	".htm":  func(p string) DocumentLoader { return NewHTMLLoader(p) },
}

// DirectoryLoaderOption configures a DirectoryLoader.
type DirectoryLoaderOption func(*DirectoryLoader)

// WithGlob sets a glob pattern to filter files (e.g. "**/*.txt").
// Only files whose base name matches the pattern are loaded.
func WithGlob(pattern string) DirectoryLoaderOption {
	return func(d *DirectoryLoader) {
		d.glob = pattern
	}
}

// WithRecursive controls whether subdirectories are traversed.
// Defaults to true.
func WithRecursive(recursive bool) DirectoryLoaderOption {
	return func(d *DirectoryLoader) {
		d.recursive = recursive
	}
}

// WithLoaderMapping adds or overrides loader factories for given extensions.
// ext must include the leading dot, e.g. ".pdf".
func WithLoaderMapping(ext string, factory LoaderFactory) DirectoryLoaderOption {
	return func(d *DirectoryLoader) {
		d.loaderMapping[ext] = factory
	}
}

// DirectoryLoader walks a directory and loads files using extension-appropriate loaders.
// Errors from individual files are collected and returned as a [MultiError] alongside
// any successfully loaded documents.
type DirectoryLoader struct {
	baseLoader
	dir           string
	glob          string
	recursive     bool
	loaderMapping map[string]LoaderFactory
}

// NewDirectoryLoader creates a DirectoryLoader rooted at dir.
func NewDirectoryLoader(dir string, opts ...DirectoryLoaderOption) *DirectoryLoader {
	// Copy default mapping so callers can extend it without side effects.
	mapping := make(map[string]LoaderFactory, len(defaultLoaderMapping))
	for k, v := range defaultLoaderMapping {
		mapping[k] = v
	}
	l := &DirectoryLoader{
		dir:           dir,
		recursive:     true,
		loaderMapping: mapping,
	}
	for _, opt := range opts {
		opt(l)
	}
	l.baseLoader.loader = l
	return l
}

// Load walks the directory and loads all matched files.
// Partial results are returned together with any per-file errors.
func (d *DirectoryLoader) Load(ctx context.Context) ([]*core.Document, error) {
	var docs []*core.Document
	var errs MultiError

	err := filepath.WalkDir(d.dir, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			errs = append(errs, fmt.Errorf("%s: %w", path, walkErr))
			return nil
		}
		if entry.IsDir() {
			if !d.recursive && path != d.dir {
				return filepath.SkipDir
			}
			return nil
		}

		// Check context between files.
		if err := ctx.Err(); err != nil {
			return err
		}

		// Apply glob filter if set.
		if d.glob != "" {
			matched, err := filepath.Match(d.glob, filepath.Base(path))
			if err != nil {
				errs = append(errs, fmt.Errorf("%s: glob match: %w", path, err))
				return nil
			}
			if !matched {
				return nil
			}
		}

		ext := filepath.Ext(path)
		factory, ok := d.loaderMapping[ext]
		if !ok {
			return nil // Skip unknown extensions silently.
		}

		loader := factory(path)
		fileDocs, err := loader.Load(ctx)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", path, err))
			return nil
		}
		docs = append(docs, fileDocs...)
		return nil
	})
	if err != nil {
		// Context cancellation or other fatal walk error.
		return docs, err
	}
	if len(errs) > 0 {
		return docs, errs
	}
	return docs, nil
}

// LoadAndSplit loads all files and splits resulting documents with splitter.
func (d *DirectoryLoader) LoadAndSplit(ctx context.Context, splitter textsplitters.TextSplitter) ([]*core.Document, error) {
	return d.baseLoader.LoadAndSplit(ctx, splitter)
}

// MultiError is a slice of errors returned when multiple files fail during
// a directory load. It implements the error interface.
type MultiError []error

// Error returns a combined error message.
func (m MultiError) Error() string {
	if len(m) == 1 {
		return m[0].Error()
	}
	msg := fmt.Sprintf("%d errors occurred:", len(m))
	for _, e := range m {
		msg += "\n  - " + e.Error()
	}
	return msg
}

// Ensure DirectoryLoader implements DocumentLoader.
var _ DocumentLoader = (*DirectoryLoader)(nil)
