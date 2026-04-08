package core

import "reflect"

// CloneMap returns a deep copy of a JSON-like map.
func CloneMap(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}
	cloned, _ := cloneValue(reflect.ValueOf(input)).Interface().(map[string]any)
	return cloned
}

func cloneValue(value reflect.Value) reflect.Value {
	if !value.IsValid() {
		return value
	}

	switch value.Kind() {
	case reflect.Interface:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		cloned := cloneValue(value.Elem())
		out := reflect.New(value.Type()).Elem()
		if cloned.IsValid() {
			out.Set(cloned)
		}
		return out
	case reflect.Map:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		cloned := reflect.MakeMapWithSize(value.Type(), value.Len())
		iter := value.MapRange()
		for iter.Next() {
			cloned.SetMapIndex(iter.Key(), cloneValue(iter.Value()))
		}
		return cloned
	case reflect.Slice:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		cloned := reflect.MakeSlice(value.Type(), value.Len(), value.Len())
		for i := 0; i < value.Len(); i++ {
			cloned.Index(i).Set(cloneValue(value.Index(i)))
		}
		return cloned
	case reflect.Array:
		cloned := reflect.New(value.Type()).Elem()
		for i := 0; i < value.Len(); i++ {
			cloned.Index(i).Set(cloneValue(value.Index(i)))
		}
		return cloned
	default:
		return value
	}
}
