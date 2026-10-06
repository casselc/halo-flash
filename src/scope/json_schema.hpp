#ifndef HALO_SCOPE_JSON_SCHEMA_HPP_
#define HALO_SCOPE_JSON_SCHEMA_HPP_

// The JSON Schema subset halo-scope's json_schema evaluator checks: type
// (one or a list of object, array, string, number, integer, boolean, null),
// properties, required, additionalProperties (bool or schema), items,
// minItems, maxItems, enum, const, minimum, maximum, exclusiveMinimum,
// exclusiveMaximum (numbers), minLength, maxLength (code points), pattern
// (ECMAScript, unanchored search), allOf, anyOf, oneOf and not.
// Annotation keywords (title, description, $schema, $id, default,
// examples, format, $comment) are accepted and ignored. Any other keyword is
// refused when the schema is checked, so no constraint is silently skipped.

#include <string>

#include "core/json.hpp"

namespace halo::scope {

/// Checks that `schema` uses only the supported keywords with valid values.
bool CheckJsonSchema(const json::Value& schema, std::string* err);

/// True when `doc` satisfies `schema` (which must pass CheckJsonSchema);
/// otherwise false with the first violation's path and reason in `*why`.
bool ValidateJsonSchema(const json::Value& schema, const json::Value& doc, std::string* why);

}  // namespace halo::scope

#endif  // HALO_SCOPE_JSON_SCHEMA_HPP_
