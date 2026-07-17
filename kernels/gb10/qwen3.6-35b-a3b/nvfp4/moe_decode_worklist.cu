// SPDX-License-Identifier: AGPL-3.0-only

// Unpadded routed-expert worklist builder for C<=8 NVFP4 MoE decode.
//
// Input rows have already been stably sorted expert-major by `moe_sort_by_expert`:
// `expert_offsets[e]..expert_offsets[e + 1]` indexes that sorted-route space.
// This builder partitions every live slice into groups of at most eight rows;
// it deliberately never creates Marlin-style padding rows.  A fixed-grid
// consumer can therefore use `total_groups` directly without a host-visible
// expert list or per-expert kernel launch.
//
// Worklist ABI (two u32 words per group):
//   word 0: expert id
//   word 1: (route_start << 4) | rows
// `route_start` indexes the existing sorted-route / expert-output layout and
// `rows` is in [1, 8].  The low four bits leave the representation explicit
// and make invalid zero-row descriptors easy for a consumer to reject.

#define ATLAS_DECODE_MAX_ROUTES 64u
#define ATLAS_DECODE_MAX_GROUP_ROWS 8u

extern "C" __global__ void moe_build_decode_worklist_c8(
    const int* __restrict__ expert_offsets,  // [num_experts + 1]
    unsigned int* __restrict__ worklist,     // [2 * max_groups]
    int* __restrict__ total_groups,          // [1]
    unsigned int num_experts,
    unsigned int routes
) {
    // The target path is explicitly C<=8, top-k=8.  Rejecting an oversized
    // input is safer than overflowing the fixed decode worklist allocation.
    if (threadIdx.x != 0) return;
    if (routes > ATLAS_DECODE_MAX_ROUTES) {
        total_groups[0] = -1;
        return;
    }

    unsigned int group = 0;
    for (unsigned int expert = 0; expert < num_experts; ++expert) {
        const int start = expert_offsets[expert];
        const int end = expert_offsets[expert + 1];
        if (start < 0 || end < start || static_cast<unsigned int>(end) > routes) {
            total_groups[0] = -1;
            return;
        }

        for (unsigned int route_start = static_cast<unsigned int>(start);
             route_start < static_cast<unsigned int>(end);
             route_start += ATLAS_DECODE_MAX_GROUP_ROWS) {
            const unsigned int remaining = static_cast<unsigned int>(end) - route_start;
            const unsigned int rows = remaining < ATLAS_DECODE_MAX_GROUP_ROWS
                ? remaining
                : ATLAS_DECODE_MAX_GROUP_ROWS;
            worklist[group * 2] = expert;
            worklist[group * 2 + 1] = (route_start << 4) | rows;
            ++group;
        }
    }
    total_groups[0] = static_cast<int>(group);
}
