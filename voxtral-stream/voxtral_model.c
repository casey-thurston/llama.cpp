/*
 * voxtral_model.c - ggml graph builders for the Voxtral Realtime 4B pieces.
 *
 * Phase 1D scope: adapter + single-token decoder. Encoder lands in a follow-up.
 *
 * The decoder builder mirrors voxtral.c's vox_decoder_forward() (single-token
 * path) directly: per layer, RMSNorm * weight, BF16 mul_mat for QKV, RoPE on
 * Q and K, write K/V into the externally-owned cache via view+cpy, read the
 * full cache slice [head_dim, n_kv, n_kv_heads] back, run flash_attn_ext,
 * project, residual; then RMSNorm * weight, SwiGLU FFN, residual.
 *
 * Layout invariants (matches llama.cpp's llama_kv_cache pattern):
 *   K/V cache  [head_dim*n_kv_heads=1024, max_seq] F32 (per layer)
 *   Q reshape  [head_dim, n_heads=32, n_tokens=1]   for RoPE
 *   K reshape  [head_dim, n_kv_heads=8, n_tokens=1] for RoPE then cache write
 *   For flash_attn_ext (per ggml.h:2322):
 *     Q permuted to [head_dim, n_tokens, n_heads]
 *     K read as [head_dim*n_kv_heads, n_kv] -> reshape -> permute to
 *                [head_dim, n_kv, n_kv_heads]
 *     V same as K
 *     mask [n_kv, n_tokens=1, 1, 1] F32 input, cast to F16 in-graph
 *
 * NOT YET WIRED in 1D (will land in 1E orchestrator):
 *   - Adaptive RMSNorm (h_norm *= 1 + ada_scale[layer]) on the FFN side. The
 *     graph just runs the plain RMSNorm path; ada_scale would be a per-session
 *     constant tensor created by the orchestrator from delay_tokens. For
 *     delay_tokens=0 the multiplier is identically 1, so this matches the 1D
 *     synthetic-input smoke test.
 *   - Encoder graph (incremental MHA + conv stem).
 */

#include "voxtral_model.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

/* RMSNorm with weight: cur = ggml_rms_norm(cur) * weight. ggml_rms_norm only
 * applies the normalization; the scale is a separate elementwise multiply,
 * matching llama.cpp's build_norm convention. */
static struct ggml_tensor * rms_norm_w(struct ggml_context * ctx,
                                       struct ggml_tensor * x,
                                       struct ggml_tensor * w,
                                       float eps) {
    struct ggml_tensor * y = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, y, w);
}

/* Linear projection: y = mul_mat(W, x). For our weight layout
 * (ggml ne[0]=in_features, ne[1]=out_features) and an input x with
 * ne[0]=in_features, ne[1]=n_tokens, this produces y with
 * ne[0]=out_features, ne[1]=n_tokens. */
static struct ggml_tensor * linear(struct ggml_context * ctx,
                                   struct ggml_tensor * w,
                                   struct ggml_tensor * x) {
    return ggml_mul_mat(ctx, w, x);
}

/* ------------------------------------------------------------------ */
/* Adapter graph                                                      */
/* ------------------------------------------------------------------ */

vox_adapter_graph_t vox_build_adapter_graph(
    struct ggml_context * gctx,
    const vox_weights_t * w,
    int n_pos)
{
    vox_adapter_graph_t out = {0};

    /* Input: encoder hidden states. ne[0]=enc_dim=1280, ne[1]=n_pos. */
    struct ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VOX_ENC_DIM, n_pos);
    ggml_set_name(x, "adapter.input");
    ggml_set_input(x);

    /* 4x temporal downsample via reshape: [1280, n_pos] -> [5120, n_pos/4].
     * Pure reinterpretation, no data movement -- works because the contiguous
     * dimension is enc_dim (innermost), so 4 consecutive positions in memory
     * are exactly 4 * 1280 = 5120 floats. n_pos must be a multiple of 4. */
    GGML_ASSERT(n_pos % VOX_DOWNSAMPLE == 0);
    struct ggml_tensor * x_ds = ggml_reshape_2d(gctx, x,
                                                VOX_ENC_DIM * VOX_DOWNSAMPLE,
                                                n_pos / VOX_DOWNSAMPLE);

    /* linear0: 5120 -> 3072 (no bias) */
    struct ggml_tensor * h = linear(gctx, w->adapter.linear0, x_ds);

    /* GELU */
    h = ggml_gelu(gctx, h);

    /* linear1: 3072 -> 3072 (no bias) */
    struct ggml_tensor * y = linear(gctx, w->adapter.linear1, h);
    ggml_set_name(y, "adapter.output");
    ggml_set_output(y);

    /* Build the graph */
    struct ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, y);

    out.gf = gf;
    out.input = x;
    out.output = y;
    return out;
}

/* ------------------------------------------------------------------ */
/* Decoder graph (single-token step)                                  */
/* ------------------------------------------------------------------ */

/* Build a single decoder transformer layer. `x` is the residual stream
 * coming in; the function adds the attention residual then the FFN residual
 * and returns the updated residual stream. KV cache writes are forward-
 * expanded into `gf` so they execute even though they're side-effects.
 * If `ada_scaled` is non-NULL, the FFN-side norm output is multiplied by it
 * (precomputed (1 + ada_scale[layer]) from delay_tokens). */
static struct ggml_tensor * build_decoder_layer(
    struct ggml_context * gctx,
    struct ggml_cgraph * gf,
    const vox_dec_layer_w_t * lw,
    const vox_kv_cache_layer_t * cache,
    struct ggml_tensor * x,
    struct ggml_tensor * pos,
    struct ggml_tensor * mask,
    struct ggml_tensor * ada_scaled, /* may be NULL */
    int n_kv,
    int kv_pos,
    int n_tokens)
{
    const int dim        = VOX_DEC_DIM;
    const int n_heads    = VOX_DEC_HEADS;
    const int n_kv_heads = VOX_DEC_KV_HEADS;
    const int head_dim   = VOX_DEC_HEAD_DIM;
    const int q_dim      = n_heads    * head_dim; /* 4096 */
    const int kv_dim     = n_kv_heads * head_dim; /* 1024 */
    const float kq_scale = 1.0f / sqrtf((float) head_dim);

    /* ---- attention ---- */
    struct ggml_tensor * x_norm = rms_norm_w(gctx, x, lw->attention_norm, VOX_DEC_NORM_EPS);

    /* QKV projections (no bias in decoder). x_norm shape [dim, n_tokens=1]. */
    struct ggml_tensor * Qcur = linear(gctx, lw->wq, x_norm); /* [q_dim, 1] */
    struct ggml_tensor * Kcur = linear(gctx, lw->wk, x_norm); /* [kv_dim, 1] */
    struct ggml_tensor * Vcur = linear(gctx, lw->wv, x_norm); /* [kv_dim, 1] */

    /* Reshape into multi-head form. ggml_rope_ext expects positions at ne[2]
     * of the input, so the reshape order is [head_dim, n_heads, n_tokens]. */
    Qcur = ggml_reshape_3d(gctx, Qcur, head_dim, n_heads,    n_tokens);
    Kcur = ggml_reshape_3d(gctx, Kcur, head_dim, n_kv_heads, n_tokens);
    Vcur = ggml_reshape_3d(gctx, Vcur, head_dim, n_kv_heads, n_tokens);

    /* Apply RoPE to Q and K. mode=0 = GGML_ROPE_TYPE_NORMAL = interleaved
     * (matches voxtral.c -- Mistral safetensors stores Q/K in interleaved
     * pairs format and the model uses non-NeoX RoPE). */
    Qcur = ggml_rope_ext(gctx, Qcur, pos, NULL,
                         head_dim,        /* n_dims (rotate the full head_dim) */
                         0,               /* mode = NORMAL/interleaved */
                         0,               /* n_ctx_orig (unused for non-YaRN) */
                         VOX_ROPE_THETA,  /* freq_base */
                         1.0f,            /* freq_scale */
                         0.0f,            /* ext_factor */
                         1.0f,            /* attn_factor */
                         32.0f,           /* beta_fast */
                         1.0f);           /* beta_slow */
    Kcur = ggml_rope_ext(gctx, Kcur, pos, NULL,
                         head_dim, 0, 0,
                         VOX_ROPE_THETA, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    /* Write the new K and V into the externally-owned cache at position kv_pos.
     *
     * Cache layout: [kv_dim=head_dim*n_kv_heads, max_seq] F32 per layer.
     *   nb[0] = sizeof(float)
     *   nb[1] = kv_dim * sizeof(float)   (stride between sequence positions)
     *
     * We write a 1D slice of `kv_dim * n_tokens` elements at byte offset
     * `kv_pos * nb[1]`. The source K/V (currently 3D after reshape) needs to
     * be flattened to 1D for the cpy. ggml_cpy treats both source and dest
     * as flat byte buffers of equal size, so we view them as 1D tensors of
     * the same nelement count. */
    const size_t cache_row_bytes = cache->k->nb[1];

    struct ggml_tensor * k_flat = ggml_reshape_1d(gctx, Kcur, kv_dim * n_tokens);
    struct ggml_tensor * v_flat = ggml_reshape_1d(gctx, Vcur, kv_dim * n_tokens);

    struct ggml_tensor * k_dst = ggml_view_1d(gctx, cache->k,
        kv_dim * n_tokens, (size_t) kv_pos * cache_row_bytes);
    struct ggml_tensor * v_dst = ggml_view_1d(gctx, cache->v,
        kv_dim * n_tokens, (size_t) kv_pos * cache_row_bytes);

    struct ggml_tensor * k_cpy = ggml_cpy(gctx, k_flat, k_dst);
    struct ggml_tensor * v_cpy = ggml_cpy(gctx, v_flat, v_dst);
    ggml_build_forward_expand(gf, k_cpy);
    ggml_build_forward_expand(gf, v_cpy);

    /* Read the full cache slice for attention. View as 2D [kv_dim, n_kv],
     * reshape to 3D [head_dim, n_kv_heads, n_kv], permute to
     * [head_dim, n_kv, n_kv_heads] (the layout flash_attn_ext expects). */
    struct ggml_tensor * K_view = ggml_view_2d(gctx, cache->k,
        kv_dim, n_kv, cache_row_bytes, 0);
    struct ggml_tensor * V_view = ggml_view_2d(gctx, cache->v,
        kv_dim, n_kv, cache_row_bytes, 0);

    struct ggml_tensor * K_3d = ggml_reshape_3d(gctx, K_view, head_dim, n_kv_heads, n_kv);
    struct ggml_tensor * V_3d = ggml_reshape_3d(gctx, V_view, head_dim, n_kv_heads, n_kv);

    struct ggml_tensor * K_perm = ggml_permute(gctx, K_3d, 0, 2, 1, 3); /* [head_dim, n_kv, n_kv_heads] */
    struct ggml_tensor * V_perm = ggml_permute(gctx, V_3d, 0, 2, 1, 3);

    /* Permute Q from [head_dim, n_heads, n_tokens] to [head_dim, n_tokens, n_heads]. */
    struct ggml_tensor * Q_perm = ggml_permute(gctx, Qcur, 0, 2, 1, 3);

    /* flash_attn_ext handles F16 K/V natively (KV cache is F16). */
    struct ggml_tensor * attn = ggml_flash_attn_ext(
        gctx, Q_perm, K_perm, V_perm, mask,
        kq_scale,
        0.0f,   /* max_bias (no ALiBi) */
        0.0f);  /* logit_softcap (none) */
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);

    /* attn shape after flash_attn_ext: [head_dim, n_heads, n_tokens] (permuted
     * per the ggml.h comment). Reshape to [q_dim, n_tokens] for the output
     * projection. The data layout is in head-major order, so the linear
     * reshape gives [head_dim*n_heads, n_tokens] correctly. */
    attn = ggml_reshape_2d(gctx, attn, q_dim, n_tokens);

    struct ggml_tensor * proj = linear(gctx, lw->wo, attn); /* [dim, 1] */

    /* Residual */
    x = ggml_add(gctx, x, proj);

    /* ---- FFN ---- */
    struct ggml_tensor * h = rms_norm_w(gctx, x, lw->ffn_norm, VOX_DEC_NORM_EPS);

    /* Adaptive RMSNorm: h *= (1 + ada_scale[layer]) on the FFN side only.
     * The orchestrator passes precomputed (1 + ada_scale[layer]) tensors so
     * the graph only needs an elementwise multiply. NULL means delay=0. */
    if (ada_scaled) {
        h = ggml_mul(gctx, h, ada_scaled);
    }

    /* SwiGLU PAR: gate = silu(w1 @ h),  up = w3 @ h,  out = w2 @ (gate * up) */
    struct ggml_tensor * gate = linear(gctx, lw->w1, h);     /* [hidden, 1] */
    gate = ggml_silu(gctx, gate);
    struct ggml_tensor * up = linear(gctx, lw->w3, h);       /* [hidden, 1] */
    struct ggml_tensor * gu = ggml_mul(gctx, gate, up);
    struct ggml_tensor * ffn = linear(gctx, lw->w2, gu);     /* [dim, 1] */

    x = ggml_add(gctx, x, ffn);

    return x;
}

vox_decoder_graph_t vox_build_decoder_graph(
    struct ggml_context * gctx,
    const vox_weights_t * w,
    const vox_kv_cache_layer_t * kv,
    struct ggml_tensor * const * ada_scaled,
    int n_kv,
    int kv_pos,
    int n_tokens)
{
    vox_decoder_graph_t out = {0};

    /* Inputs */
    struct ggml_tensor * input = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VOX_DEC_DIM, n_tokens);
    ggml_set_name(input, "decoder.input");
    ggml_set_input(input);

    struct ggml_tensor * pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(pos, "decoder.pos");
    ggml_set_input(pos);

    /* Mask: [n_kv, n_tokens, 1, 1] F32 input. */
    struct ggml_tensor * mask = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, n_kv, n_tokens, 1, 1);
    ggml_set_name(mask, "decoder.mask");
    ggml_set_input(mask);

    struct ggml_tensor * mask_f16 = ggml_cast(gctx, mask, GGML_TYPE_F16);

    /* Build the cgraph -- need it before the per-layer builders because each
     * layer forward-expands its KV cache cpy nodes. */
    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx,
        /* size  = */ 8192,
        /* grads = */ false);

    /* Walk all 26 layers */
    struct ggml_tensor * x = input;
    for (int i = 0; i < VOX_DEC_LAYERS; i++) {
        struct ggml_tensor * ada_i = ada_scaled ? ada_scaled[i] : NULL;
        x = build_decoder_layer(gctx, gf,
                                &w->decoder.layers[i],
                                &kv[i],
                                x, pos, mask_f16, ada_i,
                                n_kv, kv_pos, n_tokens);
    }

    /* Final RMSNorm + tied LM head */
    x = rms_norm_w(gctx, x, w->decoder.norm, VOX_DEC_NORM_EPS);
    struct ggml_tensor * logits = linear(gctx, w->decoder.tok_embeddings, x); /* [vocab, n_tokens] */
    ggml_set_name(logits, "decoder.logits");
    ggml_set_output(logits);

    /* GPU-side argmax — avoids reading 512KB of logits back to CPU. */
    struct ggml_tensor * argmax_t = ggml_argmax(gctx, logits); /* [n_tokens] i32 */
    ggml_set_name(argmax_t, "decoder.argmax");
    ggml_set_output(argmax_t);

    ggml_build_forward_expand(gf, argmax_t);

    out.gf     = gf;
    out.input  = input;
    out.pos    = pos;
    out.mask   = mask;
    out.logits = logits;
    out.argmax = argmax_t;
    return out;
}

/* ------------------------------------------------------------------ */
/* Encoder graph (offline, full sequence)                             */
/* ------------------------------------------------------------------ */

/* Encoder transformer layer. Differences from the decoder layer:
 *   - biases on wq/wv/wo and w2 (wk/w1/w3 have no bias)
 *   - full MHA: n_kv_heads == n_heads, head_dim=64
 *   - no KV cache, no ada_scale, no streaming -- offline path
 *   - the K/V we compute are also the K/V we attend to (self-attention) */
static struct ggml_tensor * build_encoder_layer(
    struct ggml_context * gctx,
    const vox_enc_layer_w_t * lw,
    struct ggml_tensor * x,
    struct ggml_tensor * pos,
    struct ggml_tensor * mask,
    int n_pos)
{
    const int dim      = VOX_ENC_DIM;
    const int n_heads  = VOX_ENC_HEADS;       /* 32 */
    const int head_dim = VOX_ENC_HEAD_DIM;    /* 64 */
    const float kq_scale = 1.0f / sqrtf((float) head_dim);

    /* ---- attention ---- */
    struct ggml_tensor * x_norm = rms_norm_w(gctx, x, lw->attention_norm, VOX_ENC_NORM_EPS);

    /* QKV projections (encoder has biases on q/v/o, none on k). */
    struct ggml_tensor * Qcur = linear(gctx, lw->wq, x_norm);
    Qcur = ggml_add(gctx, Qcur, lw->wq_bias);

    struct ggml_tensor * Kcur = linear(gctx, lw->wk, x_norm);
    /* wk has NO bias */

    struct ggml_tensor * Vcur = linear(gctx, lw->wv, x_norm);
    Vcur = ggml_add(gctx, Vcur, lw->wv_bias);

    /* Reshape for RoPE: [head_dim, n_heads, n_pos]. Full MHA so K/V also use n_heads. */
    Qcur = ggml_reshape_3d(gctx, Qcur, head_dim, n_heads, n_pos);
    Kcur = ggml_reshape_3d(gctx, Kcur, head_dim, n_heads, n_pos);
    Vcur = ggml_reshape_3d(gctx, Vcur, head_dim, n_heads, n_pos);

    /* Interleaved RoPE on Q and K (mode=0 = NORMAL). */
    Qcur = ggml_rope_ext(gctx, Qcur, pos, NULL,
                         head_dim, 0, 0,
                         VOX_ROPE_THETA, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    Kcur = ggml_rope_ext(gctx, Kcur, pos, NULL,
                         head_dim, 0, 0,
                         VOX_ROPE_THETA, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    /* For self-attention with the full sequence in one shot, K and V ARE
     * Kcur/Vcur (no cache). Permute to [head_dim, n_pos, n_heads] (the
     * flash_attn_ext layout). */
    struct ggml_tensor * Q_perm = ggml_permute(gctx, Qcur, 0, 2, 1, 3);
    struct ggml_tensor * K_perm = ggml_permute(gctx, Kcur, 0, 2, 1, 3);
    struct ggml_tensor * V_perm = ggml_permute(gctx, Vcur, 0, 2, 1, 3);

    /* Make K and V contiguous (cont) before passing to flash_attn_ext --
     * permutes are non-contiguous views. Q can stay as a view since
     * flash_attn handles q permutations directly per llama.cpp. */
    /* Make K and V contiguous (cont) before passing to flash_attn_ext —
     * permutes are non-contiguous views and fattn needs contiguous K/V
     * for the offline (non-cache) encoder path. */
    struct ggml_tensor * K_cont = ggml_cont(gctx, K_perm);
    struct ggml_tensor * V_cont = ggml_cont(gctx, V_perm);

    struct ggml_tensor * attn = ggml_flash_attn_ext(
        gctx, Q_perm, K_cont, V_cont, mask,
        kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);

    /* Reshape result [head_dim, n_heads, n_pos] -> [q_dim, n_pos]. */
    attn = ggml_reshape_2d(gctx, attn, head_dim * n_heads, n_pos);

    /* Output projection + bias + residual. */
    struct ggml_tensor * proj = linear(gctx, lw->wo, attn);
    proj = ggml_add(gctx, proj, lw->wo_bias);

    x = ggml_add(gctx, x, proj);

    /* ---- FFN ---- */
    struct ggml_tensor * h = rms_norm_w(gctx, x, lw->ffn_norm, VOX_ENC_NORM_EPS);

    /* SwiGLU: gate = silu(w1 @ h), up = w3 @ h, ffn = w2 @ (gate*up) + w2_bias */
    struct ggml_tensor * gate = linear(gctx, lw->w1, h);
    gate = ggml_silu(gctx, gate);
    struct ggml_tensor * up = linear(gctx, lw->w3, h);
    struct ggml_tensor * gu = ggml_mul(gctx, gate, up);
    struct ggml_tensor * ffn = linear(gctx, lw->w2, gu);
    ffn = ggml_add(gctx, ffn, lw->w2_bias);

    x = ggml_add(gctx, x, ffn);

    return x;
}

/* Encoder transformer layer with KV cache (incremental step). Shape variant
 * of build_encoder_layer with these differences:
 *   - Q is computed for n_new positions only
 *   - K/V are computed for n_new positions, written to the cache at offset
 *     kv_pos via view+cpy (mirroring the decoder pattern)
 *   - K/V for attention come from the cache slice [0, n_total)
 *   - Mask shape is [n_total, n_new, 1, 1] (caller fills it) */
static struct ggml_tensor * build_encoder_step_layer(
    struct ggml_context * gctx,
    struct ggml_cgraph * gf,
    const vox_enc_layer_w_t * lw,
    const vox_enc_kv_layer_t * cache,
    struct ggml_tensor * x,
    struct ggml_tensor * pos,
    struct ggml_tensor * mask,
    int n_new,
    int n_total,
    int kv_pos)
{
    const int dim       = VOX_ENC_DIM;
    const int n_heads   = VOX_ENC_HEADS;       /* 32 */
    const int head_dim  = VOX_ENC_HEAD_DIM;    /* 64 */
    const int q_dim     = n_heads * head_dim;  /* 2048 */
    const int kv_dim    = q_dim;               /* full MHA */
    const float kq_scale = 1.0f / sqrtf((float) head_dim);

    /* ---- attention ---- */
    struct ggml_tensor * x_norm = rms_norm_w(gctx, x, lw->attention_norm, VOX_ENC_NORM_EPS);

    /* QKV projections (encoder has biases on q/v/o, none on k). */
    struct ggml_tensor * Qcur = linear(gctx, lw->wq, x_norm); /* [q_dim, n_new] */
    Qcur = ggml_add(gctx, Qcur, lw->wq_bias);

    struct ggml_tensor * Kcur = linear(gctx, lw->wk, x_norm); /* [kv_dim, n_new] */

    struct ggml_tensor * Vcur = linear(gctx, lw->wv, x_norm); /* [kv_dim, n_new] */
    Vcur = ggml_add(gctx, Vcur, lw->wv_bias);

    /* Reshape into multi-head form. ggml_rope_ext expects positions at ne[2]. */
    Qcur = ggml_reshape_3d(gctx, Qcur, head_dim, n_heads, n_new);
    Kcur = ggml_reshape_3d(gctx, Kcur, head_dim, n_heads, n_new);
    Vcur = ggml_reshape_3d(gctx, Vcur, head_dim, n_heads, n_new);

    /* Interleaved RoPE on Q and K (mode=0). pos has absolute encoder positions. */
    Qcur = ggml_rope_ext(gctx, Qcur, pos, NULL,
                         head_dim, 0, 0,
                         VOX_ROPE_THETA, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    Kcur = ggml_rope_ext(gctx, Kcur, pos, NULL,
                         head_dim, 0, 0,
                         VOX_ROPE_THETA, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    /* Write the new K and V into the cache at byte offset kv_pos * row_bytes.
     * Cache row stride is the same as the decoder cache pattern. */
    const size_t cache_row_bytes = cache->k->nb[1];

    struct ggml_tensor * k_flat = ggml_reshape_1d(gctx, Kcur, kv_dim * n_new);
    struct ggml_tensor * v_flat = ggml_reshape_1d(gctx, Vcur, kv_dim * n_new);

    struct ggml_tensor * k_dst = ggml_view_1d(gctx, cache->k,
        kv_dim * n_new, (size_t) kv_pos * cache_row_bytes);
    struct ggml_tensor * v_dst = ggml_view_1d(gctx, cache->v,
        kv_dim * n_new, (size_t) kv_pos * cache_row_bytes);

    struct ggml_tensor * k_cpy = ggml_cpy(gctx, k_flat, k_dst);
    struct ggml_tensor * v_cpy = ggml_cpy(gctx, v_flat, v_dst);
    ggml_build_forward_expand(gf, k_cpy);
    ggml_build_forward_expand(gf, v_cpy);

    /* Read the full cache slice [0, n_total) for attention. */
    struct ggml_tensor * K_view = ggml_view_2d(gctx, cache->k,
        kv_dim, n_total, cache_row_bytes, 0);
    struct ggml_tensor * V_view = ggml_view_2d(gctx, cache->v,
        kv_dim, n_total, cache_row_bytes, 0);

    struct ggml_tensor * K_3d = ggml_reshape_3d(gctx, K_view, head_dim, n_heads, n_total);
    struct ggml_tensor * V_3d = ggml_reshape_3d(gctx, V_view, head_dim, n_heads, n_total);

    struct ggml_tensor * K_perm = ggml_permute(gctx, K_3d, 0, 2, 1, 3); /* [head_dim, n_total, n_heads] */
    struct ggml_tensor * V_perm = ggml_permute(gctx, V_3d, 0, 2, 1, 3);

    /* Permute Q from [head_dim, n_heads, n_new] to [head_dim, n_new, n_heads]. */
    struct ggml_tensor * Q_perm = ggml_permute(gctx, Qcur, 0, 2, 1, 3);

    /* flash_attn_ext handles F16 K/V natively (KV cache is F16). */
    struct ggml_tensor * attn = ggml_flash_attn_ext(
        gctx, Q_perm, K_perm, V_perm, mask,
        kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);

    /* Reshape result [head_dim, n_heads, n_new] -> [q_dim, n_new]. */
    attn = ggml_reshape_2d(gctx, attn, q_dim, n_new);

    /* Output projection + bias + residual. */
    struct ggml_tensor * proj = linear(gctx, lw->wo, attn);
    proj = ggml_add(gctx, proj, lw->wo_bias);

    x = ggml_add(gctx, x, proj);

    /* ---- FFN (same as offline encoder layer) ---- */
    struct ggml_tensor * h = rms_norm_w(gctx, x, lw->ffn_norm, VOX_ENC_NORM_EPS);

    struct ggml_tensor * gate = linear(gctx, lw->w1, h);
    gate = ggml_silu(gctx, gate);
    struct ggml_tensor * up = linear(gctx, lw->w3, h);
    struct ggml_tensor * gu = ggml_mul(gctx, gate, up);
    struct ggml_tensor * ffn = linear(gctx, lw->w2, gu);
    ffn = ggml_add(gctx, ffn, lw->w2_bias);

    x = ggml_add(gctx, x, ffn);

    return x;
}

vox_encoder_step_graph_t vox_build_encoder_step_graph(
    struct ggml_context * gctx,
    const vox_weights_t * w,
    const vox_enc_kv_layer_t * kv,
    int n_new,
    int n_total,
    int kv_pos)
{
    vox_encoder_step_graph_t out = {0};

    struct ggml_tensor * input = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VOX_ENC_DIM, n_new);
    ggml_set_name(input, "encoder_step.input");
    ggml_set_input(input);

    /* Mask: rows = new query positions (n_new), cols = cache positions (n_total). */
    struct ggml_tensor * mask = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, n_total, n_new, 1, 1);
    ggml_set_name(mask, "encoder_step.mask");
    ggml_set_input(mask);

    struct ggml_tensor * mask_f16 = ggml_cast(gctx, mask, GGML_TYPE_F16);

    struct ggml_tensor * pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_new);
    ggml_set_name(pos, "encoder_step.pos");
    ggml_set_input(pos);

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 8192, false);

    struct ggml_tensor * x = input;
    for (int i = 0; i < VOX_ENC_LAYERS; i++) {
        x = build_encoder_step_layer(gctx, gf, &w->encoder.layers[i], &kv[i],
                                     x, pos, mask_f16, n_new, n_total, kv_pos);
    }

    x = rms_norm_w(gctx, x, w->encoder.norm, VOX_ENC_NORM_EPS);
    ggml_set_name(x, "encoder_step.output");
    ggml_set_output(x);

    ggml_build_forward_expand(gf, x);

    out.gf = gf;
    out.input = input;
    out.pos = pos;
    out.mask = mask;
    out.output = x;
    return out;
}

vox_encoder_graph_t vox_build_encoder_graph(
    struct ggml_context * gctx,
    const vox_weights_t * w,
    int n_pos)
{
    vox_encoder_graph_t out = {0};

    /* Inputs */
    struct ggml_tensor * input = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, VOX_ENC_DIM, n_pos);
    ggml_set_name(input, "encoder.input");
    ggml_set_input(input);

    /* Causal sliding-window mask: [n_pos, n_pos, 1, 1] F32. The orchestrator
     * fills this with 0 where j <= i and -INF where j > i (or beyond window). */
    struct ggml_tensor * mask = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, n_pos, n_pos, 1, 1);
    ggml_set_name(mask, "encoder.mask");
    ggml_set_input(mask);

    struct ggml_tensor * mask_f16 = ggml_cast(gctx, mask, GGML_TYPE_F16);

    /* Position vector for RoPE: [n_pos] i32 with values 0..n_pos-1. */
    struct ggml_tensor * pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos, "encoder.pos");
    ggml_set_input(pos);

    /* Build the cgraph */
    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 8192, false);

    /* 32 transformer layers */
    struct ggml_tensor * x = input;
    for (int i = 0; i < VOX_ENC_LAYERS; i++) {
        x = build_encoder_layer(gctx, &w->encoder.layers[i], x, pos, mask_f16, n_pos);
    }

    /* Final RMSNorm * encoder.norm */
    x = rms_norm_w(gctx, x, w->encoder.norm, VOX_ENC_NORM_EPS);
    ggml_set_name(x, "encoder.output");
    ggml_set_output(x);

    ggml_build_forward_expand(gf, x);

    out.gf = gf;
    out.input = input;
    out.pos = pos;
    out.mask = mask;
    out.output = x;
    return out;
}

