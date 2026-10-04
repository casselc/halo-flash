// P0 microbenchmark kernels, compiled once to a gfx1151 HSACO and loaded by
// every backend (ROCm HIP, native HRX, HRX HIP shim) so they run identical
// machine code.
#define KERNEL [[clang::amdgpu_kernel, gnu::visibility("protected")]]

typedef unsigned int u32;
typedef unsigned long u64;
typedef float float4_t __attribute__((ext_vector_type(4)));

KERNEL void halo_noop(void) {}

// Streams `n4` float4 elements; each workgroup writes one partial sum so the
// loads cannot be dead-code eliminated. Grid-stride over a fixed grid.
KERNEL void halo_read_bw(const float4_t* restrict src, float* restrict partial,
                         u64 n4) {
  const u32 lid = __builtin_amdgcn_workitem_id_x();
  const u32 gid = __builtin_amdgcn_workgroup_id_x();
  const u32 wg = __builtin_amdgcn_workgroup_size_x();
  const u64 stride = (u64)__builtin_amdgcn_grid_size_x();
  float4_t acc = {0, 0, 0, 0};
  for (u64 i = (u64)gid * wg + lid; i < n4; i += stride) {
    acc += __builtin_nontemporal_load(&src[i]);
  }
  float s = acc.x + acc.y + acc.z + acc.w;
  // A data-dependent store that is never taken in practice.
  if (s == 1234567.0f) partial[gid] = s;
}

KERNEL void halo_write_bw(float4_t* restrict dst, u64 n4) {
  const u32 lid = __builtin_amdgcn_workitem_id_x();
  const u32 gid = __builtin_amdgcn_workgroup_id_x();
  const u32 wg = __builtin_amdgcn_workgroup_size_x();
  const u64 stride = (u64)__builtin_amdgcn_grid_size_x();
  const float4_t v = {1, 2, 3, 4};
  for (u64 i = (u64)gid * wg + lid; i < n4; i += stride) {
    __builtin_nontemporal_store(v, &dst[i]);
  }
}

// Writes `value` to out[0]; used to check argument passing on each backend.
KERNEL void halo_store(u32* out, u32 value) { out[0] = value; }
