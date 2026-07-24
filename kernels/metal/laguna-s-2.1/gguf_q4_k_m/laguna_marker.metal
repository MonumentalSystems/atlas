// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

kernel void laguna_marker(uint gid [[thread_position_in_grid]]) {
    (void)gid;
}
