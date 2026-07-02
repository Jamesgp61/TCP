#ifndef TELEMETRY_PACK_H
#define TELEMETRY_PACK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Topology constants ---- */
#define TOPOLOGY_BITS         6
#define TOPOLOGY_NODE_COUNT   (1u << TOPOLOGY_BITS)   /* 64 nodes  */
#define TOPOLOGY_DEGREE       TOPOLOGY_BITS            /* 6 neighbors (Hamming-1) */

/* ---- Node identifier: 6-bit value + 2 flag bits ---- */
typedef struct {
    uint8_t id;       /* lower 6 bits used, range [0,63] */
    uint8_t flags;    /* upper 2 bits: reserved / parity / ack */
} node_id_t;

/* ---- Fixed-point Q8.8: signed 16-bit, 8 integer / 8 fractional ---- */
typedef int16_t q8_8_t;

#define Q8_8_SCALE       256
#define Q8_8_FROM_FLOAT(f)  ((q8_8_t)((f) * 256.0f))
#define Q8_8_TO_FLOAT(q)    ((float)(q) / 256.0f)

/* ---- Telemetry frame (matches socket payload schema) ----
 * Wire layout, little-endian, 19 bytes total:
 *   [0]       node_id(6) | flags(2)
 *   [1..2]    state_energy   (q8_8, signed)
 *   [3..6]    timestamp      (uint32)
 *   [7..18]   edge_weights[6] (q8_8, signed), indexed by bit position 0..5
 */
typedef struct {
    node_id_t  node;
    q8_8_t     state_energy;
    uint32_t   timestamp;
    q8_8_t     edge_weights[TOPOLOGY_DEGREE];
} telemetry_frame_t;

#define TELEMETRY_FRAME_SIZE  19u

/* ---- Serialization ---- */
size_t telemetry_pack(const telemetry_frame_t *frame,
                      uint8_t *buf, size_t buf_len);

size_t telemetry_unpack(telemetry_frame_t *frame,
                        const uint8_t *buf, size_t buf_len);

/* ---- Topology helpers ---- */
uint8_t telemetry_neighbor_id(uint8_t node_id, uint8_t bit_index);
uint8_t telemetry_hamming_distance(uint8_t a, uint8_t b);

#ifdef __cplusplus
}
#endif

#endif /* TELEMETRY_PACK_H */