/* halo-flash variants (PLAN.md P10): load, verify and select variants of a
 * base model from C. A registry belongs to one base model; loading a
 * variant whose manifest names another base content hash is refused.
 *
 *   char err[512];
 *   halo_variant_registry* r = halo_variant_registry_open(&base_path, 1, err, sizeof err);
 *   halo_variant_load(r, "variants/sentiment-ab/manifest.json", name, sizeof name, err, sizeof err);
 *   ... engines take the variant by name (Qwen35Gpu::AddAdapter, the decider,
 *       the server's "variant" field / "base:variant" model alias) ...
 *   halo_variant_registry_close(r);
 *
 * Functions return 0 on success and -1 on failure with a NUL-terminated
 * message in err (truncated to errlen). Thread-safe. */
#ifndef HALO_VARIANT_H_
#define HALO_VARIANT_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct halo_variant_registry halo_variant_registry;

/* Opens the base model (one GGUF file, or the shards of a split model in
 * order) and computes its content hash (cached in "<file>.halo-sha256"). */
halo_variant_registry* halo_variant_registry_open(const char* const* base_files, size_t n_files, char* err,
                                                  size_t errlen);
void halo_variant_registry_close(halo_variant_registry* r);

/* The base content hash (64 hex digits), valid until close. */
const char* halo_variant_registry_base_hash(const halo_variant_registry* r);

/* Loads and verifies a manifest (format, base hash, part sizes and SHA-256,
 * LoRA shapes). On success the variant's name is copied to name_out. */
int halo_variant_load(halo_variant_registry* r, const char* manifest_path, char* name_out, size_t name_len,
                      char* err, size_t errlen);
/* Returns 0 if the variant was loaded and is now unloaded, -1 otherwise. */
int halo_variant_unload(halo_variant_registry* r, const char* name);

/* Number of loaded variants, and the i-th name (sorted) copied to out. */
size_t halo_variant_count(const halo_variant_registry* r);
int halo_variant_name(const halo_variant_registry* r, size_t i, char* out, size_t out_len);

/* The loaded manifest as JSON (including its eval summary). Returns the
 * length the JSON needs (excluding the NUL); writes it if it fits in
 * buf_len, else writes nothing. -1 if no such variant. */
long halo_variant_manifest_json(const halo_variant_registry* r, const char* name, char* buf, size_t buf_len);

/* Splits an OpenAI model value "base:variant" (no ':' = base only).
 * Returns 0 and copies the parts, or -1 for an invalid variant name. */
int halo_variant_parse_model_alias(const char* model, char* base_out, size_t base_len, char* variant_out,
                                   size_t variant_len);

#ifdef __cplusplus
}
#endif

#endif /* HALO_VARIANT_H_ */
