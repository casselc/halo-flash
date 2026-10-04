#!/usr/bin/env python3
"""Per-kernel resource report from an HSACO's AMDGPU metadata notes.

For every kernel: VGPRs, SGPRs, spills (VGPR/SGPR), scratch (private segment)
bytes, LDS (group segment) bytes, and an occupancy estimate in waves per
SIMD for gfx1151 wave32 (1536 VGPRs per SIMD lane slot, allocated in granules
of 24 -- LLVM's getVGPRAllocGranule for Feature1_5xVGPRs parts; at most 16
waves; LDS 128 KiB per WGP of 4 SIMDs, allocated in 512-byte granules). It
matches the compiler's own -Rpass-analysis=kernel-resource-usage occupancy
for VGPR-limited kernels. The metadata .vgpr_count is the exact count used,
and rocprofv3's arch_vgpr_count rounds to 8 (the descriptor encoding), so
neither shows the 24-register allocation. The estimate is a reading aid, not
a measurement.

  kernel_report.py --hsaco build/gpu/gen/core.hsaco [--readelf PATH] [--filter REGEX]
  kernel_report.py --hsaco ... --check tools/kernel_spill_allowlist.txt

--check exits 1 when a kernel spills or uses scratch and is not listed in
the allowlist (one kernel name per line, '#' comments), or when a listed
kernel no longer spills (so the list stays honest).

  kernel_report.py --compile HIPCC --source core.hip --include src [--readelf PATH]

--compile recompiles the source with -Rpass-analysis=kernel-resource-usage
and exits 1 unless this tool's waves/SIMD estimate equals the compiler's
own occupancy for every kernel of that compile (it keeps the estimate
honest: the VGPR granule was once 16 here).
"""

import argparse
import math
import re
import shutil
import subprocess
import sys
import tempfile


def readelf_path(arg):
    if arg:
        return arg
    for name in ("llvm-readelf", "readelf"):
        p = shutil.which(name)
        if p:
            return p
    sys.exit("llvm-readelf not found; pass --readelf")


def parse(notes):
    kernels, cur = [], None
    for line in notes.splitlines():
        m = re.match(r"\s*-?\s*\.(\w+):\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if key == "args":
            continue
        if key == "name" and val.startswith("halo_"):
            cur = {"name": val}
            kernels.append(cur)
        elif cur is not None and key in (
                "vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count",
                "private_segment_fixed_size", "group_segment_fixed_size", "max_flat_workgroup_size",
                "wavefront_size"):
            try:
                cur[key] = int(val)
            except ValueError:
                pass
    # Metadata lists fields alphabetically: fields before .name belong to the
    # same kernel map, so read the notes map by map instead when possible.
    return kernels


def parse_maps(notes):
    """Splits the kernels list into maps (each starting at '- .'), robust to field order."""
    out = []
    in_kernels = False
    block = []
    for line in notes.splitlines():
        if re.match(r"^amdhsa\.kernels:", line.strip()):
            in_kernels = True
            continue
        if not in_kernels:
            continue
        if re.match(r"^amdhsa\.\w+:", line.strip()):
            break
        if re.match(r"^\s{2}- \.", line):
            if block:
                out.append(block)
            block = [line]
        elif block:
            block.append(line)
    if block:
        out.append(block)
    kernels = []
    for b in out:
        k = {}
        depth = None
        for line in b:
            m = re.match(r"^(\s*)(?:- )?\.(\w+):\s*(.*)$", line)
            if not m:
                continue
            indent = len(m.group(1)) + (2 if "- ." in line else 0)
            if depth is None:
                depth = indent
            if indent != depth:
                continue  # nested (.args entries)
            key, val = m.group(2), m.group(3).strip()
            if key == "name":
                k["name"] = val
            else:
                try:
                    k[key] = int(val)
                except ValueError:
                    pass
        if "name" in k:
            kernels.append(k)
    return kernels


def occupancy(k):
    v = max(k.get("vgpr_count", 0), 1)
    granule = 24
    by_vgpr = min(16, 1536 // (math.ceil(v / granule) * granule))
    lds = k.get("group_segment_fixed_size", 0)
    wg = max(k.get("max_flat_workgroup_size", 256), 32)
    waves_per_block = math.ceil(wg / 32)
    if lds > 0:
        blocks_per_wgp = (128 * 1024) // (math.ceil(lds / 512) * 512)
        by_lds = max(1, (blocks_per_wgp * waves_per_block) // 4)
        return min(by_vgpr, by_lds), ("lds" if by_lds < by_vgpr else "vgpr")
    return by_vgpr, "vgpr"


def compiler_occupancy(remarks):
    occ, cur = {}, None
    for m in re.finditer(r"remark:\s+([A-Za-z \[\]/]+?): (\S+) \[", remarks):
        key, val = m.groups()
        if key == "Function Name":
            cur = val
        elif key.startswith("Occupancy") and cur:
            occ[cur] = int(val)
    return occ


def compare_with_compiler(args):
    if not args.source:
        sys.exit("--compile needs --source")
    with tempfile.TemporaryDirectory() as tmp:
        hsaco = f"{tmp}/k.hsaco"
        cmd = [args.compile, "-x", "hip", "--offload-arch=gfx1151", "--offload-device-only", "--no-gpu-bundle-output",
               "-O3", "-std=c++20", "-Rpass-analysis=kernel-resource-usage", "-o", hsaco, args.source]
        cmd += [f"-I{i}" for i in args.include]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr[-4000:])
        notes = subprocess.run([readelf_path(args.readelf), "--notes", hsaco], check=True,
                               capture_output=True, text=True).stdout
    kernels = parse_maps(notes)
    occ = compiler_occupancy(r.stderr)
    bad = []
    for k in kernels:
        ours, lim = occupancy(k)
        theirs = occ.get(k["name"])
        if theirs != ours:
            bad.append(f"{k['name']}: estimate {ours} ({lim}), compiler {theirs} "
                       f"(vgpr {k.get('vgpr_count', 0)}, lds {k.get('group_segment_fixed_size', 0)}, "
                       f"wg {k.get('max_flat_workgroup_size', 0)})")
    for b in bad:
        print(b)
    print(f"{len(kernels)} kernels, {len(occ)} compiler occupancies, {len(bad)} mismatches")
    return 1 if bad or not kernels else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsaco")
    ap.add_argument("--compile", metavar="HIPCC", help="recompile --source and compare occupancy with the compiler's")
    ap.add_argument("--source")
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--readelf")
    ap.add_argument("--filter", default="")
    ap.add_argument("--check", metavar="ALLOWLIST")
    args = ap.parse_args()
    if args.compile:
        return compare_with_compiler(args)
    if not args.hsaco:
        ap.error("--hsaco is required")
    notes = subprocess.run([readelf_path(args.readelf), "--notes", args.hsaco], check=True,
                           capture_output=True, text=True).stdout
    kernels = parse_maps(notes)
    if not kernels:
        sys.exit("no kernels found in the notes")
    rx = re.compile(args.filter)
    if args.check:
        allowed = set()
        with open(args.check) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    allowed.add(line)
        bad = []
        spilling = set()
        for k in kernels:
            spills = k.get("vgpr_spill_count", 0) + k.get("sgpr_spill_count", 0)
            if spills or k.get("private_segment_fixed_size", 0):
                spilling.add(k["name"])
                if k["name"] not in allowed:
                    bad.append(f"{k['name']}: {k.get('vgpr_spill_count', 0)} VGPR / {k.get('sgpr_spill_count', 0)} "
                               f"SGPR spills, {k.get('private_segment_fixed_size', 0)} B scratch")
        stale = sorted(allowed - spilling)
        for s in stale:
            bad.append(f"{s}: listed in the allowlist but no longer spills or uses scratch; remove it")
        for b in bad:
            print(b)
        print(f"{len(kernels)} kernels, {len(spilling)} with spills or scratch, {len(bad)} problems")
        return 1 if bad else 0
    print(f"{'kernel':42} {'vgpr':>4} {'sgpr':>4} {'vspill':>6} {'scratch':>7} {'lds':>6} {'wg':>5} {'waves/SIMD':>10}")
    for k in sorted(kernels, key=lambda k: k["name"]):
        if not rx.search(k["name"]):
            continue
        occ, lim = occupancy(k)
        print(f"{k['name']:42} {k.get('vgpr_count', 0):4} {k.get('sgpr_count', 0):4} {k.get('vgpr_spill_count', 0):6} "
              f"{k.get('private_segment_fixed_size', 0):7} {k.get('group_segment_fixed_size', 0):6} "
              f"{k.get('max_flat_workgroup_size', 0):5} {occ:7} ({lim})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
