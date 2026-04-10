/*
 * voxtral_model.h - ggml graph builders for the Voxtral Realtime 4B pieces.
 *
 * One builder per piece. Each builder takes a graph context, the loaded
 * weights, and the per-call inputs (input tensors, KV cache tensors,
 * positions); it constructs ggml ops in the context, expands the result into
 * a freshly created cgraph, and returns the (cgraph, output_tensor) pair so
 * the caller can allocate intermediate storage and execute.
 *
 * Phase 1D scope:
 *   - vox_build_adapter_graph: 5 ops (downsample reshape + Linear + GELU + Linear)
 *   - vox_build_decoder_graph: single-token decode step against an externally-
 *                              owned per-layer KV cache. ada_scale is OMITTED
 *                              for now (TODO: wire from the orchestrator in 1E).
 *
 * Encoder graph builder lands in a follow-up commit; structure mirrors the
 * decoder layer with biases on q/v/o/w2 and full MHA instead of GQA.
 */

#ifndef VOXTRAL_MODEL_H
#define VOXTRAL_MODEL_H

#include "ggml.h"
#include "voxtral_weights.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Architectural constants. Must match voxtral.h. */
#define VOX_DEC_DIM        3072
#define VOX_DEC_HEADS      32
#define VOX_DEC_KV_HEADS   8
#define VOX_DEC_HEAD_DIM   128
#define VOX_DEC_HIDDEN     9216
#define VOX_DEC_WINDOW     8192
#define VOX_DEC_NORM_EPS   1e-5f
#define VOX_VOCAB_SIZE     131072
#define VOX_ROPE_THETA     1000000.0f

#define VOX_ENC_DIM        1280
#define VOX_ENC_HEADS      32
#define VOX_ENC_KV_HEADS   32   /* full MHA, not GQA */
#define VOX_ENC_HEAD_DIM   64
#define VOX_ENC_HIDDEN     5120
#define VOX_ENC_WINDOW     750
#define VOX_ENC_NORM_EPS   1e-5f
#define VOX_MEL_BINS       128

#define VOX_DOWNSAMPLE     4

/* ------------------------------------------------------------------ */
/* Adapter                                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    struct ggml_cgraph  * gf;        /* the computed graph */
    struct ggml_tensor  * input;     /* [enc_dim, n_pos] -- f32, set by caller */
    struct ggml_tensor  * output;    /* [dec_dim, n_pos / VOX_DOWNSAMPLE] -- f32 */
} vox_adapter_graph_t;

/* Build the adapter graph for `n_pos` encoder positions (must be a multiple
 * of VOX_DOWNSAMPLE = 4). Output has n_pos/4 positions in dec_dim space.
 *
 *   reshape [enc_dim=1280, n_pos] -> [enc_dim*4=5120, n_pos/4]
 *   linear0 (5120 -> 3072)
 *   GELU
 *   linear1 (3072 -> 3072)
 */
vox_adapter_graph_t vox_build_adapter_graph(
    struct ggml_context * gctx,
    const vox_weights_t * w,
    int n_pos);

/* ------------------------------------------------------------------ */
/* Decoder (single-token step)                                        */
/* ------------------------------------------------------------------ */

/* Per-layer KV cache, externally owned. Each tensor has shape
 *   [head_dim * n_kv_heads, max_seq]
 * with type GGML_TYPE_F32 (matches llama.cpp's KV cache layout). Writes
 * happen as 1D slices at position kv_pos; reads happen as 2D slices then
 * reshape+permute into the [head_dim, n_kv, n_kv_heads] layout that
 * flash_attn_ext expects. */
typedef struct {
    struct ggml_tensor * k;          /* f32 [head_dim*n_kv_heads, max_seq] */
    struct ggml_tensor * v;          /* f32 [head_dim*n_kv_heads, max_seq] */
} vox_kv_cache_layer_t;

#define VOX_DEC_LAYERS_HDR 26 /* duplicated from voxtral_weights.h to keep this header self-contained */

typedef struct {
    struct ggml_cgraph * gf;
    struct ggml_tensor * input;      /* f32 [dec_dim, 1] -- the new step's input embedding */
    struct ggml_tensor * pos;        /* i32 [1] -- logical RoPE position for this step */
    struct ggml_tensor * mask;       /* f32 [n_kv_padded, 1] -- attention mask (0 or -INF) */
    struct ggml_tensor * logits;     /* f32 [vocab] -- output logits over the tied vocab */
} vox_decoder_graph_t;

/* Build the single-token decoder step graph.
 *
 *   - n_kv: total cache occupancy AFTER this step (i.e., previous + 1).
 *           Used to slice the K/V views to [head_dim, n_kv, n_kv_heads].
 *   - kv_pos: physical write position for the new step (== n_kv - 1).
 *             Used to compute byte offsets into the cache views for ggml_cpy.
 *   - kv: per-layer KV cache tensors (length VOX_DEC_LAYERS_HDR).
 *   - ada_scaled: per-layer (1 + ada_scale) F32 [dec_dim] precomputed by the
 *                 orchestrator from delay_tokens. Pass NULL to skip the
 *                 multiplication entirely (equivalent to delay_tokens=0).
 *
 * Caller is responsible for filling input/pos/mask via ggml_backend_tensor_set
 * before calling ggml_backend_graph_compute. */
vox_decoder_graph_t vox_build_decoder_graph(
    struct ggml_context * gctx,
    const vox_weights_t * w,
    const vox_kv_cache_layer_t * kv, /* length VOX_DEC_LAYERS_HDR */
    struct ggml_tensor * const * ada_scaled, /* length VOX_DEC_LAYERS_HDR or NULL */
    int n_kv,
    int kv_pos);

/* ------------------------------------------------------------------ */
/* Encoder (full sequence in one graph -- offline path)               */
/* ------------------------------------------------------------------ */

typedef struct {
    struct ggml_cgraph * gf;
    struct ggml_tensor * input;   /* f32 [enc_dim, n_pos] -- post-conv-stem hidden */
    struct ggml_tensor * pos;     /* i32 [n_pos] -- RoPE positions, 0..n_pos-1 */
    struct ggml_tensor * mask;    /* f32 [n_pos, n_pos, 1, 1] -- causal sliding-window */
    struct ggml_tensor * output;  /* f32 [enc_dim, n_pos] -- final encoder hidden */
} vox_encoder_graph_t;

/* Build the offline encoder graph for n_pos positions (the conv stem runs in
 * plain C in the orchestrator and produces this n_pos-length sequence). The
 * graph runs all 32 transformer layers + the final RMSNorm. */
vox_encoder_graph_t vox_build_encoder_graph(
    struct ggml_context * gctx,
    const vox_weights_t * w,
    int n_pos);

#ifdef __cplusplus
}
#endif

#endif /* VOXTRAL_MODEL_H */
