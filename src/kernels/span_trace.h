// Span tracing: per-wave phase timestamps inside kernels (after Sites's KUtrace:
// record every phase transition cheaply, rebuild timelines afterwards; see
// tools/span_report.py).
//
// Compiled in only with -DHALO_TRACE, into a separate code object
// (core_trace.hsaco, CMake target `core_trace`); in the engine's own build
// every macro expands to nothing.
//
//   HALO_SPAN_BEGIN(kernel_tag, max_events)  once per wave at kernel start:
//       reserves max_events records for this wave and records event 0
//   HALO_SPAN(event)                          records `event` (1..255)
//
// A record is 16 bytes, written by lane 0 of the wave: the shader cycle
// counter (20 bits) and where the wave runs (SE, SA, WGP, SIMD from HW_ID1),
// the block's linear index, wave-in-block, the event and the kernel tag; wave
// Every traced wave also writes an anchor record (event 255: the 100 MHz
// realtime clock read with the cycles) after its start stamp, and wave 0 one at
// the end: SHADER_CYCLES is per SIMD in the engine, so each wave is placed by
// its own anchor; wave 0's pair gives the block's clock (tools/span_report.py).
// An event costs one s_getreg and one 16-byte store. The host clears and
// reads halo_spans through Device::WriteGlobal / ReadGlobal.
#ifndef HALO_KERNELS_SPAN_TRACE_H_
#define HALO_KERNELS_SPAN_TRACE_H_

#include <cstdint>

// Kernel tags (tools/span_report.py names them).
enum HaloSpanTag : std::uint8_t {
  kSpanControl = 1,      // halo_span_control: known delays, for checking the tracer
  kSpanCombineNorm = 2,  // halo_hc_combine_norm_rows
  kSpanGemmQ8rF16 = 3,   // GemmQ8rF16 (every kernel built on it)
  kSpanMoeDown = 4,      // MoeDown (halo_moe_down_grouped_*)
  kSpanMoeGateUp = 5,    // MoeGateUp (halo_moe_gate_up_grouped_*)
  kSpanQsel = 6,         // halo_attention_wmma_qsel_h
};

#ifdef HALO_TRACE

struct HaloSpan {
  std::uint32_t realtime;  // 100 MHz; anchor records (event 255) and HALO_SPAN_RT only, else 0
  std::uint32_t cycles;    // SHADER_CYCLES in bits 0-19; where the wave ran in bits 20-31 (see below)
  std::uint32_t block;     // linear block index
  std::uint8_t wave, event, tag, valid;
};
static_assert(sizeof(HaloSpan) == 16);
typedef std::uint32_t halo_trace_v4u __attribute__((ext_vector_type(4)));

// Each tag owns a quarter of the buffer; a wave's records sit at
// (block * waves per block + wave) * max_events within it: no atomics, and a
// later launch of the same tag overwrites (trace one launch per tag).
constexpr std::uint32_t kHaloSpanCap = 1u << 22;  // 4 Mi records, 64 MiB
constexpr std::uint32_t kHaloSpanRegion = kHaloSpanCap / 4;
__device__ HaloSpan halo_spans[kHaloSpanCap];
// Blocks with block % halo_span_sample == 0 record (host-set, default every block); the others run the
// same code without storing, so sampling lowers the overhead the traced blocks see from their neighbours.
__device__ std::uint32_t halo_span_sample = 1;
__device__ std::uint32_t halo_span_offset = 0;  // records blocks with block % sample == offset
// 0 records nothing. The engine (halo_run --span-*) clears it after loading and brackets the one dispatch it
// traces with halo_span_set(1) / halo_span_set(0), stream-ordered kernels: no host synchronization.
__device__ std::uint32_t halo_span_enable = 1;
// Traced waves whose slots fell past their tag's region (recorded nothing): the host reports it, so a
// truncated trace is never silent (a 16660-block MoE grid overflowed at block ~5700 before slots counted
// sampled blocks only).
__device__ std::uint32_t halo_span_dropped = 0;

struct HaloSpanState {
  std::uint32_t base, n, max, hw;
  std::uint32_t block;
  std::uint8_t wave, tag;
};

constexpr std::uint32_t kHaloSpanAnchor = 255;  // event id of an anchor record (realtime + cycles read together)

__device__ inline void HaloSpanStore(HaloSpanState& s, std::uint32_t event, std::uint32_t realtime, std::uint32_t cy) {
  if (s.n < s.max && (threadIdx.x % 32) == 0) {
    // Stored through inline asm: the compiler's wait insertion does not track it, so a barrier's release
    // fence (s_waitcnt_vscnt 0 before s_barrier) does not wait for trace records to reach memory. With a
    // plain store every traced barrier paid a store round trip (GEMM +37%, docs/TRACING.md).
    const std::uint32_t w3 = s.wave | (event << 8) | (std::uint32_t{s.tag} << 16) | (1u << 24);
    const halo_trace_v4u r = {realtime, cy | s.hw, s.block, w3};
    HaloSpan* dst = &halo_spans[s.base + s.n];
    asm volatile("global_store_b128 %0, %1, off" : : "v"(dst), "v"(r));
  }
  ++s.n;
}

__device__ inline void HaloSpanWrite(HaloSpanState& s, std::uint32_t event, std::uint32_t realtime = 0) {
  HaloSpanStore(s, event, realtime, static_cast<std::uint32_t>(clock64()) & 0xFFFFFu);
}

__device__ inline HaloSpanState HaloSpanBegin(std::uint32_t tag, std::uint32_t max_events) {
  // The start stamp comes first: the slot and sampling arithmetic below loads globals (a memory round trip,
  // ~3-4 us under a saturated DRAM queue), which made wave 0, the one that also checks for an anchor, start late.
  const std::uint32_t cy0 = static_cast<std::uint32_t>(clock64()) & 0xFFFFFu;
  // Read in uniform control flow so they are scalar loads (inside the lane-0 branch they became vector loads
  // waited on at once: ~1 us per wave).
  const std::uint32_t sample = halo_span_sample, offset = halo_span_offset, enable = halo_span_enable;
  HaloSpanState s;
  const std::uint32_t tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const std::uint32_t wpb = (blockDim.x * blockDim.y * blockDim.z + 31) / 32;
  s.block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  // Wave-uniform state in scalar registers (readfirstlane): in vector registers it raised the mix GEMM's
  // VGPRs 209 -> 242 and cut its occupancy 7 -> 5 waves/SIMD (+40% time with recording off).
  s.wave = static_cast<std::uint8_t>(__builtin_amdgcn_readfirstlane(tid / 32));
  s.tag = static_cast<std::uint8_t>(tag);
  s.n = 0;
  // Slots count sampled blocks only (block / sample), so sampling multiplies the capacity.
  const std::uint64_t slot = (std::uint64_t{s.block / sample} * wpb + s.wave) * max_events;
  s.base = (tag % 4) * kHaloSpanRegion + static_cast<std::uint32_t>(slot);
  const bool sampled = enable && s.block % sample == offset;
  const bool fits = slot + max_events <= kHaloSpanRegion;
  if (sampled && !fits && (tid % 32) == 0) atomicAdd(&halo_span_dropped, 1u);
  s.max = sampled && fits ? max_events : 0;
  s.base = __builtin_amdgcn_readfirstlane(s.base);
  s.max = __builtin_amdgcn_readfirstlane(s.max);
  // HW_ID1 (gfx11): SIMD_ID 9:8, WGP_ID 13:10, SA_ID 16, SE_ID 20:18 -> bits 20-31 as
  // SIMD 2 bits, WGP 4, SA 1, SE 3.
  std::uint32_t hw;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw));
  s.hw = __builtin_amdgcn_readfirstlane((((hw >> 8) & 3u) | (((hw >> 10) & 15u) << 2) | (((hw >> 16) & 1u) << 6) | (((hw >> 18) & 7u) << 7)) << 20);
  HaloSpanStore(s, 0, 0, cy0);
  // SHADER_CYCLES is effectively per SIMD: under continuous load the SIMDs' counters agree to tens of cycles,
  // but in the engine (after idle power-downs) SIMDs of one WGP differ by arbitrary offsets. So every traced
  // wave reads the realtime clock (a ~0.6 us message, after its start stamp) into an anchor record with the
  // cycles read right after it; wave 0 adds a closing anchor (HALO_SPAN_END) for the block's clock.
  if (s.max) {
    const std::uint32_t rt = static_cast<std::uint32_t>(wall_clock64());
    HaloSpanStore(s, kHaloSpanAnchor, rt, static_cast<std::uint32_t>(clock64()) & 0xFFFFFu);
  }
  return s;
}

__device__ inline void HaloSpanEnd(HaloSpanState& s, std::uint32_t event) {
  HaloSpanWrite(s, event);
  if (s.wave == 0 && s.max) {
    const std::uint32_t rt = static_cast<std::uint32_t>(wall_clock64());
    HaloSpanStore(s, kHaloSpanAnchor, rt, static_cast<std::uint32_t>(clock64()) & 0xFFFFFu);
  }
}

#define HALO_SPAN_BEGIN(tag, max_events) HaloSpanState halo_span_ = HaloSpanBegin((tag), (max_events))
#define HALO_SPAN(event) HaloSpanWrite(halo_span_, (event))
// The kernel's last event: also writes wave 0's closing anchor.
#define HALO_SPAN_END(event) HaloSpanEnd(halo_span_, (event))
// Also records the realtime clock (a ~0.6 us message), read just before the cycles: for checking times.
#define HALO_SPAN_RT(event) HaloSpanWrite(halo_span_, (event), static_cast<std::uint32_t>(wall_clock64()))

#else

#define HALO_SPAN_BEGIN(tag, max_events) \
  do {                                   \
  } while (0)
#define HALO_SPAN(event) \
  do {                   \
  } while (0)
#define HALO_SPAN_RT(event) \
  do {                      \
  } while (0)
#define HALO_SPAN_END(event) \
  do {                       \
  } while (0)

#endif  // HALO_TRACE

#endif  // HALO_KERNELS_SPAN_TRACE_H_
