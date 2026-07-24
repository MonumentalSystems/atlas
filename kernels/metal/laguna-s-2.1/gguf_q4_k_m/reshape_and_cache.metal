// SPDX-License-Identifier: AGPL-3.0-only

#include <metal_stdlib>
using namespace metal;

kernel void reshape_and_cache_flash(
    device const bfloat *key [[buffer(0)]],
    device const bfloat *value [[buffer(1)]],
    device bfloat *k_cache [[buffer(2)]],
    device bfloat *v_cache [[buffer(3)]],
    device const long *slot_mapping [[buffer(4)]],
    constant uint &num_kv_heads [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &block_size [[buffer(7)]],
    constant uint &key_stride [[buffer(8)]],
    constant uint &value_stride [[buffer(9)]],
    constant uint &physical_blocks [[buffer(10)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    const long slot = slot_mapping[token];
    if (slot < 0 || physical_blocks == 0) {
        return;
    }
    const ulong logical_block = ulong(slot) / block_size;
    const ulong block = logical_block % physical_blocks;
    const ulong block_offset = ulong(slot) % block_size;
    const uint elements = num_kv_heads * head_dim;
    const ulong destination = (block * block_size + block_offset) * elements;
    const ulong key_source = ulong(token) * key_stride;
    const ulong value_source = ulong(token) * value_stride;
    for (uint index = tid; index < elements; index += threads) {
        k_cache[destination + index] = key[key_source + index];
        v_cache[destination + index] = value[value_source + index];
    }
}
