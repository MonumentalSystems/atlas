// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

kernel void fill_slots_from_block_table(
    device long *slots [[buffer(0)]],
    device const uint *block_table [[buffer(1)]],
    constant uint &start_pos [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &block_size [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) return;
    const uint position = start_pos + index;
    const uint logical_block = position / block_size;
    slots[index] = long(ulong(block_table[logical_block]) * block_size
        + position % block_size);
}
