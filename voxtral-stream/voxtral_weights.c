/*
 * voxtral_weights.c - Load all 711 Voxtral Realtime 4B safetensors into a
 * single ggml backend buffer.
 *
 * Algorithm:
 *   1. Open consolidated.safetensors via the vendored mmap reader.
 *   2. Create a no_alloc ggml_context.
 *   3. Per-slot pass: for each named weight, find it in safetensors, allocate
 *      a ggml_tensor of the right shape+dtype, store the (src, dst) record.
 *   4. After all slots are populated, allocate one contiguous backend buffer
 *      via ggml_backend_alloc_ctx_tensors_from_buft.
 *   5. Copy pass: for each record, push raw BF16 bytes (zero-copy from mmap)
 *      or BF16->F32-converted bytes into the corresponding tensor via
 *      ggml_backend_tensor_set (which handles host->device for CUDA).
 *   6. Close the safetensors file (mmap region no longer needed).
 *
 * Error model: any missing tensor, shape mismatch, or backend allocation
 * failure prints a loud error to stderr, frees what's already allocated, and
 * returns NULL. The smoke test in main.c will exit non-zero if anything is
 * wrong.
 */

#include "voxtral_weights.h"
#include "voxtral_safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Architectural constants -- must match voxtral.c/voxtral.h. */
#define ENC_PREFIX "mm_streams_embeddings.embedding_module.whisper_encoder"
#define ADAPTER_LINEAR0 "mm_streams_embeddings.embedding_module.audio_language_projection.0.weight"
#define ADAPTER_LINEAR1 "mm_streams_embeddings.embedding_module.audio_language_projection.2.weight"
#define TOK_EMBED_NAME  "mm_streams_embeddings.embedding_module.tok_embeddings.weight"
#define DEC_FINAL_NORM  "norm.weight"

/* Upper bound on the number of tracked weight records (711 actual + slack). */
#define WLOAD_MAX 800

typedef struct {
    const safetensor_t * src;
    struct ggml_tensor * dst;
    int                  to_f32; /* 1 = convert BF16->F32 at copy, 0 = direct BF16 */
} wload_record_t;

typedef struct {
    safetensors_file_t * sf;
    struct ggml_context * ctx;
    wload_record_t records[WLOAD_MAX];
    int n;
    int err;
} wload_state_t;

/* ------------------------------------------------------------------------ */
/* Per-slot helpers                                                         */
/* ------------------------------------------------------------------------ */

static int sf_numel(const safetensor_t * t) {
    int64_t n = 1;
    for (int d = 0; d < t->ndim; d++) n *= t->shape[d];
    return (int) n;
}

/* Look up a tensor by name. On miss, log + flag the state and return NULL. */
static const safetensor_t * wload_find(wload_state_t * s, const char * name) {
    const safetensor_t * t = safetensors_find(s->sf, name);
    if (!t) {
        fprintf(stderr, "vox_weights: missing tensor: %s\n", name);
        s->err = 1;
    }
    return t;
}

/* Push a record into the state's record table. Caller has already created the
 * ggml tensor and verified the dtype/shape. */
static void wload_push(wload_state_t * s, const safetensor_t * src,
                       struct ggml_tensor * dst, int to_f32) {
    if (s->n >= WLOAD_MAX) {
        fprintf(stderr, "vox_weights: WLOAD_MAX exceeded (%d)\n", WLOAD_MAX);
        s->err = 1;
        return;
    }
    s->records[s->n].src = src;
    s->records[s->n].dst = dst;
    s->records[s->n].to_f32 = to_f32;
    s->n++;
}

/* Verify that a ggml tensor we're about to fill matches the safetensor's
 * element count. Logs the mismatch and flags state on failure. */
static int wload_check_numel(wload_state_t * s, const char * name,
                             struct ggml_tensor * dst, const safetensor_t * src) {
    int64_t want = ggml_nelements(dst);
    int64_t got  = sf_numel(src);
    if (want != got) {
        fprintf(stderr, "vox_weights: nelem mismatch for %s: ggml=%lld safetensors=%lld\n",
                name, (long long) want, (long long) got);
        s->err = 1;
        return -1;
    }
    return 0;
}

/* Load a 1D F32 tensor (norm or bias). Source is BF16 in safetensors;
 * safetensors_get_f32 will convert at copy time. */
static struct ggml_tensor * wload_f32_1d(wload_state_t * s, const char * name) {
    const safetensor_t * src = wload_find(s, name);
    if (!src) return NULL;
    if (src->ndim != 1) {
        fprintf(stderr, "vox_weights: %s: expected 1D, got %dD\n", name, src->ndim);
        s->err = 1;
        return NULL;
    }
    struct ggml_tensor * dst = ggml_new_tensor_1d(s->ctx, GGML_TYPE_F32, src->shape[0]);
    ggml_set_name(dst, name);
    if (wload_check_numel(s, name, dst, src) != 0) return NULL;
    wload_push(s, src, dst, 1);
    return dst;
}

/* Load a 2D F32 tensor (e.g. ada_norm_down/up). Stored row-major in safetensors
 * as [out, in]; ggml wants ne[0]=in, ne[1]=out (innermost first). */
static struct ggml_tensor * wload_f32_2d(wload_state_t * s, const char * name) {
    const safetensor_t * src = wload_find(s, name);
    if (!src) return NULL;
    if (src->ndim != 2) {
        fprintf(stderr, "vox_weights: %s: expected 2D, got %dD\n", name, src->ndim);
        s->err = 1;
        return NULL;
    }
    struct ggml_tensor * dst = ggml_new_tensor_2d(s->ctx, GGML_TYPE_F32,
                                                  src->shape[1], src->shape[0]);
    ggml_set_name(dst, name);
    if (wload_check_numel(s, name, dst, src) != 0) return NULL;
    wload_push(s, src, dst, 1);
    return dst;
}

/* Load a 3D F32 tensor (conv stem). Safetensors layout for a PyTorch conv1d
 * weight is [out_channels, in_channels, kernel]; ggml innermost-first becomes
 * ne[0]=kernel, ne[1]=in_channels, ne[2]=out_channels. */
static struct ggml_tensor * wload_f32_3d(wload_state_t * s, const char * name) {
    const safetensor_t * src = wload_find(s, name);
    if (!src) return NULL;
    if (src->ndim != 3) {
        fprintf(stderr, "vox_weights: %s: expected 3D, got %dD\n", name, src->ndim);
        s->err = 1;
        return NULL;
    }
    struct ggml_tensor * dst = ggml_new_tensor_3d(s->ctx, GGML_TYPE_F32,
                                                  src->shape[2], src->shape[1], src->shape[0]);
    ggml_set_name(dst, name);
    if (wload_check_numel(s, name, dst, src) != 0) return NULL;
    wload_push(s, src, dst, 1);
    return dst;
}

/* Load a 2D BF16 matmul weight (zero-copy from mmap region). */
static struct ggml_tensor * wload_bf16_2d(wload_state_t * s, const char * name) {
    const safetensor_t * src = wload_find(s, name);
    if (!src) return NULL;
    if (src->ndim != 2) {
        fprintf(stderr, "vox_weights: %s: expected 2D, got %dD\n", name, src->ndim);
        s->err = 1;
        return NULL;
    }
    if (src->dtype != DTYPE_BF16) {
        fprintf(stderr, "vox_weights: %s: expected bf16, got dtype=%d\n", name, src->dtype);
        s->err = 1;
        return NULL;
    }
    struct ggml_tensor * dst = ggml_new_tensor_2d(s->ctx, GGML_TYPE_BF16,
                                                  src->shape[1], src->shape[0]);
    ggml_set_name(dst, name);
    if (wload_check_numel(s, name, dst, src) != 0) return NULL;
    wload_push(s, src, dst, 0);
    return dst;
}

/* ------------------------------------------------------------------------ */
/* Public API                                                               */
/* ------------------------------------------------------------------------ */

vox_weights_t * vox_weights_load(const char * model_dir, ggml_backend_buffer_type_t buft) {
    /* ---- 1. Open safetensors file ---- */
    char path[1024];
    snprintf(path, sizeof(path), "%s/consolidated.safetensors", model_dir);

    safetensors_file_t * sf = safetensors_open(path);
    if (!sf) {
        fprintf(stderr, "vox_weights_load: cannot open %s\n", path);
        return NULL;
    }

    /* ---- 2. Allocate the result struct + ggml context (no_alloc) ---- */
    vox_weights_t * w = (vox_weights_t *) calloc(1, sizeof(*w));
    if (!w) {
        safetensors_close(sf);
        return NULL;
    }

    struct ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (sf->num_tensors + 64),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    w->ctx = ggml_init(ip);
    if (!w->ctx) {
        fprintf(stderr, "vox_weights_load: ggml_init failed\n");
        free(w);
        safetensors_close(sf);
        return NULL;
    }

    /* ---- 3. Per-slot tensor creation pass ---- */
    wload_state_t s = {0};
    s.sf = sf;
    s.ctx = w->ctx;

    char name[512];

    /* Encoder conv stem (F32, 3D for weights, 1D for biases) */
    snprintf(name, sizeof(name), "%s.conv_layers.0.conv.weight", ENC_PREFIX);
    w->encoder.conv0 = wload_f32_3d(&s, name);
    snprintf(name, sizeof(name), "%s.conv_layers.0.conv.bias", ENC_PREFIX);
    w->encoder.conv0_bias = wload_f32_1d(&s, name);
    snprintf(name, sizeof(name), "%s.conv_layers.1.conv.weight", ENC_PREFIX);
    w->encoder.conv1 = wload_f32_3d(&s, name);
    snprintf(name, sizeof(name), "%s.conv_layers.1.conv.bias", ENC_PREFIX);
    w->encoder.conv1_bias = wload_f32_1d(&s, name);

    /* Encoder transformer layers */
    for (int i = 0; i < VOX_ENC_LAYERS; i++) {
        vox_enc_layer_w_t * l = &w->encoder.layers[i];

        snprintf(name, sizeof(name), "%s.transformer.layers.%d.attention.wq.weight", ENC_PREFIX, i);
        l->wq = wload_bf16_2d(&s, name);
        snprintf(name, sizeof(name), "%s.transformer.layers.%d.attention.wq.bias", ENC_PREFIX, i);
        l->wq_bias = wload_f32_1d(&s, name);
        snprintf(name, sizeof(name), "%s.transformer.layers.%d.attention.wk.weight", ENC_PREFIX, i);
        l->wk = wload_bf16_2d(&s, name);
        /* wk has NO bias */
        snprintf(name, sizeof(name), "%s.transformer.layers.%d.attention.wv.weight", ENC_PREFIX, i);
        l->wv = wload_bf16_2d(&s, name);
        snprintf(name, sizeof(name), "%s.transformer.layers.%d.attention.wv.bias", ENC_PREFIX, i);
        l->wv_bias = wload_f32_1d(&s, name);
        snprintf(name, sizeof(name), "%s.transformer.layers.%d.attention.wo.weight", ENC_PREFIX, i);
        l->wo = wload_bf16_2d(&s, name);
        snprintf(name, sizeof(name), "%s.transformer.layers.%d.attention.wo.bias", ENC_PREFIX, i);
        l->wo_bias = wload_f32_1d(&s, name);
        snprintf(name, sizeof(name), "%s.transformer.layers.%d.attention_norm.weight", ENC_PREFIX, i);
        l->attention_norm = wload_f32_1d(&s, name);

        snprintf(name, sizeof(name), "%s.transformer.layers.%d.feed_forward.w1.weight", ENC_PREFIX, i);
        l->w1 = wload_bf16_2d(&s, name);
        snprintf(name, sizeof(name), "%s.transformer.layers.%d.feed_forward.w2.weight", ENC_PREFIX, i);
        l->w2 = wload_bf16_2d(&s, name);
        snprintf(name, sizeof(name), "%s.transformer.layers.%d.feed_forward.w2.bias", ENC_PREFIX, i);
        l->w2_bias = wload_f32_1d(&s, name);
        snprintf(name, sizeof(name), "%s.transformer.layers.%d.feed_forward.w3.weight", ENC_PREFIX, i);
        l->w3 = wload_bf16_2d(&s, name);
        snprintf(name, sizeof(name), "%s.transformer.layers.%d.ffn_norm.weight", ENC_PREFIX, i);
        l->ffn_norm = wload_f32_1d(&s, name);
    }

    /* Encoder final norm */
    snprintf(name, sizeof(name), "%s.transformer.norm.weight", ENC_PREFIX);
    w->encoder.norm = wload_f32_1d(&s, name);

    /* Adapter (BF16 linears, no biases) */
    w->adapter.linear0 = wload_bf16_2d(&s, ADAPTER_LINEAR0);
    w->adapter.linear1 = wload_bf16_2d(&s, ADAPTER_LINEAR1);

    /* Decoder tied embedding */
    w->decoder.tok_embeddings = wload_bf16_2d(&s, TOK_EMBED_NAME);

    /* Decoder transformer layers */
    for (int i = 0; i < VOX_DEC_LAYERS; i++) {
        vox_dec_layer_w_t * l = &w->decoder.layers[i];

        snprintf(name, sizeof(name), "layers.%d.ada_rms_norm_t_cond.0.weight", i);
        l->ada_norm_down = wload_f32_2d(&s, name);
        snprintf(name, sizeof(name), "layers.%d.ada_rms_norm_t_cond.2.weight", i);
        l->ada_norm_up = wload_f32_2d(&s, name);

        snprintf(name, sizeof(name), "layers.%d.attention.wq.weight", i);
        l->wq = wload_bf16_2d(&s, name);
        snprintf(name, sizeof(name), "layers.%d.attention.wk.weight", i);
        l->wk = wload_bf16_2d(&s, name);
        snprintf(name, sizeof(name), "layers.%d.attention.wv.weight", i);
        l->wv = wload_bf16_2d(&s, name);
        snprintf(name, sizeof(name), "layers.%d.attention.wo.weight", i);
        l->wo = wload_bf16_2d(&s, name);
        snprintf(name, sizeof(name), "layers.%d.attention_norm.weight", i);
        l->attention_norm = wload_f32_1d(&s, name);

        snprintf(name, sizeof(name), "layers.%d.feed_forward.w1.weight", i);
        l->w1 = wload_bf16_2d(&s, name);
        snprintf(name, sizeof(name), "layers.%d.feed_forward.w2.weight", i);
        l->w2 = wload_bf16_2d(&s, name);
        snprintf(name, sizeof(name), "layers.%d.feed_forward.w3.weight", i);
        l->w3 = wload_bf16_2d(&s, name);
        snprintf(name, sizeof(name), "layers.%d.ffn_norm.weight", i);
        l->ffn_norm = wload_f32_1d(&s, name);
    }

    /* Decoder final norm */
    w->decoder.norm = wload_f32_1d(&s, DEC_FINAL_NORM);

    if (s.err) {
        fprintf(stderr, "vox_weights_load: errors during tensor creation pass\n");
        ggml_free(w->ctx);
        free(w);
        safetensors_close(sf);
        return NULL;
    }

    /* Sanity: every record we tracked must have a non-NULL dst (it would
     * already be flagged as err if not, but double-check). */
    if (s.n != sf->num_tensors) {
        fprintf(stderr, "vox_weights_load: tracked %d records but safetensors has %d tensors -- "
                        "some safetensor names were not consumed\n", s.n, sf->num_tensors);
        /* Not fatal; log unmapped names for visibility. */
        for (int i = 0; i < sf->num_tensors; i++) {
            const safetensor_t * st = &sf->tensors[i];
            int found = 0;
            for (int j = 0; j < s.n; j++) {
                if (s.records[j].src == st) { found = 1; break; }
            }
            if (!found) fprintf(stderr, "  unmapped: %s\n", st->name);
        }
        ggml_free(w->ctx);
        free(w);
        safetensors_close(sf);
        return NULL;
    }

    /* ---- 4. Allocate the contiguous backend buffer ---- */
    w->buf = ggml_backend_alloc_ctx_tensors_from_buft(w->ctx, buft);
    if (!w->buf) {
        fprintf(stderr, "vox_weights_load: ggml_backend_alloc_ctx_tensors_from_buft failed\n");
        ggml_free(w->ctx);
        free(w);
        safetensors_close(sf);
        return NULL;
    }
    w->n_tensors = s.n;
    w->n_bytes   = ggml_backend_buffer_get_size(w->buf);

    /* ---- 5. Copy pass: push bytes into each tensor ---- */
    for (int i = 0; i < s.n; i++) {
        const safetensor_t * src = s.records[i].src;
        struct ggml_tensor * dst = s.records[i].dst;
        size_t nbytes = ggml_nbytes(dst);

        if (s.records[i].to_f32) {
            /* BF16 -> F32 conversion (allocates a fresh float[]). */
            float * f32 = safetensors_get_f32(sf, src);
            if (!f32) {
                fprintf(stderr, "vox_weights_load: safetensors_get_f32 failed for %s\n", src->name);
                vox_weights_free(w);
                safetensors_close(sf);
                return NULL;
            }
            ggml_backend_tensor_set(dst, f32, 0, nbytes);
            free(f32);
        } else {
            /* Direct BF16 zero-copy from mmap region. */
            const void * raw = safetensors_data(sf, src);
            if (!raw) {
                fprintf(stderr, "vox_weights_load: safetensors_data failed for %s\n", src->name);
                vox_weights_free(w);
                safetensors_close(sf);
                return NULL;
            }
            ggml_backend_tensor_set(dst, raw, 0, nbytes);
        }
    }

    /* ---- 6. Done with mmap region. ---- */
    safetensors_close(sf);

    return w;
}

void vox_weights_free(vox_weights_t * w) {
    if (!w) return;
    if (w->buf) ggml_backend_buffer_free(w->buf);
    if (w->ctx) ggml_free(w->ctx);
    free(w);
}
