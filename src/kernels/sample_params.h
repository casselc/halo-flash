#ifndef HALO_KERNELS_SAMPLE_PARAMS_H_
#define HALO_KERNELS_SAMPLE_PARAMS_H_

// Device-side sampling: the request's filter settings, its penalty list and
// the RNG state go to the GPU with the step; the token and the advanced RNG
// state come back. Semantics are core/sampling's (gufo's): penalties, top-k,
// temperature, top-p, min-p, min_keep floors, ties to the lower token id,
// and the same xorshift RNG, advanced only when more than one token can be
// drawn. Shared by the kernel (HIP) and the engine (host).

#include <stdint.h>

struct HaloSampleParams {
  float temperature;
  float top_p;
  float min_p;
  float repeat_penalty;
  float frequency_penalty;
  float presence_penalty;
  uint32_t top_k;  // 0: off
  uint32_t min_keep;
  uint32_t n_penalties;
  uint32_t vocab;
  uint64_t rng;  // RNG state before this draw
};

// One entry per penalized token, as core/sampling's TokenPenalty.
struct HaloPenalty {
  uint32_t token;
  uint32_t generated_count;
  uint32_t repeated;
  uint32_t pad;
};

struct HaloSampleOut {
  uint32_t token;  // 0xFFFFFFFF: no finite logit
  uint32_t live;   // kept tokens with nonzero fixed-point weight
  uint64_t rng;    // RNG state after this draw
};

// Device limits: the penalized-token bitmap lives in LDS.
#define HALO_SAMPLE_MAX_VOCAB 262144u
#define HALO_SAMPLE_MAX_PENALTIES 8192u
// Weights are fixed point (scale 2^40) so mass sums are exact and
// order-independent. A token whose weight relative to the max is below
// 2^-40 (about 1e-12) is never drawn.
#define HALO_SAMPLE_FIXED_BITS 40

#endif  // HALO_KERNELS_SAMPLE_PARAMS_H_
