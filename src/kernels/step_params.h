#ifndef HALO_KERNELS_STEP_PARAMS_H_
#define HALO_KERNELS_STEP_PARAMS_H_

// Per-step values a captured decode graph reads from device memory, so one
// recorded graph replays at any position. Shared by the kernels (HIP) and
// the engine (host); a uint32 array indexed by these slots.
enum HaloStepParam {
  kHaloToken = 0,     // input token id
  kHaloPosition,      // position of the token (0-based)
  kHaloNkv,           // keys visible to causal attention: position + 1
  kHaloComplete,      // complete QSA blocks: n_kv / ratio
  kHaloBlock,         // block holding this token: position / ratio
  kHaloNVisible,      // keys attended after QSA selection
  kHaloStepParamCount
};

#endif  // HALO_KERNELS_STEP_PARAMS_H_
