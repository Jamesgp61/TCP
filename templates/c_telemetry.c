#include "telemetry_pack.h"
#include <string.h>

/* ---- Explicit little-endian byte helpers (no endianness assumptions) ---- */
static inline void put_u16_le(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
}

static inline uint16_t get_u16_le(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static inline void put_u32_le(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static inline uint32_t get_u32_le(const uint8_t *p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

/* ---- Pack frame into raw byte buffer ---- */
size_t telemetry_pack(const telemetry_frame_t *frame,
                      uint8_t *buf, size_t buf_len) {
    if (!frame || !buf || buf_len < TELEMETRY_FRAME_SIZE) {
        return 0u;
    }

    /* [0] node_id(6) | flags(2) */
    buf[0] = (uint8_t)((frame->node.id & 0x3Fu) |
                       ((frame->node.flags & 0x03u) << 6));

    /* [1..2] state_energy */
    put_u16_le(&buf[1], (uint16_t)frame->state_energy);

    /* [3..6] timestamp */
    put_u32_le(&buf[3], frame->timestamp);

    /* [7..18] 6 edge weights */
    for (uint8_t i = 0u; i < TOPOLOGY_DEGREE; ++i) {
        put_u16_le(&buf[7u + (size_t)i * 2u],
                   (uint16_t)frame->edge_weights[i]);
    }

    return TELEMETRY_FRAME_SIZE;
}

/* ---- Unpack raw byte buffer into frame ---- */
size_t telemetry_unpack(telemetry_frame_t *frame,
                        const uint8_t *buf, size_t buf_len) {
    if (!frame || !buf || buf_len < TELEMETRY_FRAME_SIZE) {
        return 0u;
    }

    memset(frame, 0, sizeof(*frame));

    frame->node.id    = (uint8_t)(buf[0] & 0x3Fu);
    frame->node.flags = (uint8_t)((buf[0] >> 6) & 0x03u);

    frame->state_energy = (q8_8_t)get_u16_le(&buf[1]);
    frame->timestamp    = get_u32_le(&buf[3]);

    for (uint8_t i = 0u; i < TOPOLOGY_DEGREE; ++i) {
        frame->edge_weights[i] =
            (q8_8_t)get_u16_le(&buf[7u + (size_t)i * 2u]);
    }

    return TELEMETRY_FRAME_SIZE;
}

/* ---- Neighbor id via single bit flip ---- */
uint8_t telemetry_neighbor_id(uint8_t node_id, uint8_t bit_index) {
    if (bit_index >= TOPOLOGY_BITS) {
        return node_id;
    }
    return (uint8_t)(node_id ^ (uint8_t)(1u << bit_index));
}

/* ---- Hamming distance over 6-bit space ---- */
uint8_t telemetry_hamming_distance(uint8_t a, uint8_t b) {
    uint8_t x = (uint8_t)((a ^ b) & 0x3Fu);
    uint8_t d = 0u;
    while (x) {
        d += (uint8_t)(x & 1u);
        x >>= 1;
    }
    return d;
}