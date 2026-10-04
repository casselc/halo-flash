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
  uint32_t fallback_only;  // exact kernel: run only if the fast path asked
  uint32_t pad;
};

// One entry per penalized token, as core/sampling's TokenPenalty.
struct HaloPenalty {
  uint32_t token;
  uint32_t generated_count;
  uint32_t repeated;
  uint32_t pad;
};

// Fast sampler (halo_sample_blocks + halo_sample_final): distribution-exact
// rather than draw-identical. Each block of HALO_SAMPLE_BLOCK tokens keeps
// its top HALO_SAMPLE_TOPK candidates; the final kernel works on those and
// requests the exact kernel when the kept set might reach past them.
#define HALO_SAMPLE_BLOCK 1024u
#define HALO_SAMPLE_TOPK 256u
#define HALO_SAMPLE_FALLBACK 0xFFFFFFFEu

struct HaloSampleBlock {
  double mass;       // sum exp((v - max_v) / T) over the block's valid tokens
  double gumbel;     // best v / T + Gumbel noise (plain temperature sampling)
  float max_v;
  uint32_t argmax;   // ties: lower token
  uint32_t gumbel_id;
  uint32_t valid;
  uint32_t kept;     // candidates written (<= HALO_SAMPLE_TOPK)
  uint32_t pad;
  uint64_t excluded; // composite key of the best token not kept (0: none)
};

struct HaloSampleOut {
  uint32_t token;  // 0xFFFFFFFF: no finite logit; HALO_SAMPLE_FALLBACK: exact kernel pending
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
