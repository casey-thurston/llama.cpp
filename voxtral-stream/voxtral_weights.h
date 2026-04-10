/*
 * voxtral_weights.h - Voxtral Realtime 4B weight loader (safetensors -> ggml)
 *
 * Mirrors the layout of voxtral.c/voxtral.h's vox_encoder_t / vox_adapter_t /
 * vox_decoder_t but with `struct ggml_tensor *` slots instead of raw float* /
 * uint16_t* pointers. The actual data lives in a backend buffer owned by
 * vox_weights_t.
 *
 * Dtype strategy (matches voxtral.c):
 *   BF16 (mmap zero-copy): all matmul weights -- encoder/decoder wq/wk/wv/wo,
 *                          w1/w2/w3, adapter linears, tok_embeddings.
 *   F32  (BF16->F32 at load): conv0/conv1 weights+biases, all biases,
 *                             attention_norm, ffn_norm, ada_norm_down/up,
 *                             encoder/decoder final norm.
 *
 * Single-call API:
 *   w = vox_weights_load(model_dir, ggml_backend_cpu_buffer_type());  // or cuda
 *   ... use w->encoder.layers[i].wq, etc. ...
 *   vox_weights_free(w);
 */

#ifndef VOXTRAL_WEIGHTS_H
#define VOXTRAL_WEIGHTS_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of voxtral.h architectural constants. Kept local so we don't have to
 * vendor voxtral.h itself (which pulls in raw-pointer weight structs we don't
 * want). */
#define VOX_ENC_LAYERS  32
#define VOX_DEC_LAYERS  26

/* Encoder transformer layer weights (whisper-style MHA + SwiGLU FFN, with
 * biases on wq/wv/wo/w2 -- wk has NO bias). */
typedef struct {
    /* Attention */
    struct ggml_tensor * wq;             /* bf16 [in=1280, out=2048] */
    struct ggml_tensor * wq_bias;        /* f32  [2048] */
    struct ggml_tensor * wk;             /* bf16 [in=1280, out=2048] -- no bias */
    struct ggml_tensor * wv;             /* bf16 [in=1280, out=2048] */
    struct ggml_tensor * wv_bias;        /* f32  [2048] */
    struct ggml_tensor * wo;             /* bf16 [in=2048, out=1280] */
    struct ggml_tensor * wo_bias;        /* f32  [1280] */
    struct ggml_tensor * attention_norm; /* f32  [1280] */

    /* Feed-forward (SwiGLU: w1 = gate, w3 = up, w2 = down). w2 has bias. */
    struct ggml_tensor * w1;             /* bf16 [in=1280, out=5120] */
    struct ggml_tensor * w2;             /* bf16 [in=5120, out=1280] */
    struct ggml_tensor * w2_bias;        /* f32  [1280] */
    struct ggml_tensor * w3;             /* bf16 [in=1280, out=5120] */
    struct ggml_tensor * ffn_norm;       /* f32  [1280] */
} vox_enc_layer_w_t;

typedef struct {
    /* Conv stem -- causal conv1d, kept as F32 (small, ~3MB) */
    struct ggml_tensor * conv0;          /* f32  [3, 128, 1280] (k, in, out) */
    struct ggml_tensor * conv0_bias;     /* f32  [1280] */
    struct ggml_tensor * conv1;          /* f32  [3, 1280, 1280] */
    struct ggml_tensor * conv1_bias;     /* f32  [1280] */

    vox_enc_layer_w_t layers[VOX_ENC_LAYERS];

    struct ggml_tensor * norm;           /* f32  [1280] -- final encoder RMSNorm */
} vox_encoder_w_t;

/* Audio-to-language adapter: Linear -> GELU -> Linear (no biases). */
typedef struct {
    struct ggml_tensor * linear0;        /* bf16 [in=5120, out=3072] */
    struct ggml_tensor * linear1;        /* bf16 [in=3072, out=3072] */
} vox_adapter_w_t;

/* Decoder transformer layer weights (Mistral-style GQA + SwiGLU FFN, no
 * biases anywhere; adaptive RMSNorm via ada_norm_down/up MLP). */
typedef struct {
    /* Adaptive RMSNorm conditioning MLP: Linear(3072->32) -> GELU -> Linear(32->3072).
     * F32 because tiny. */
    struct ggml_tensor * ada_norm_down;  /* f32 [in=3072, out=32]   -- safetensors [32, 3072] */
    struct ggml_tensor * ada_norm_up;    /* f32 [in=32,   out=3072] -- safetensors [3072, 32] */

    /* Attention (GQA: 32 q heads / 8 kv heads / head_dim=128) */
    struct ggml_tensor * wq;             /* bf16 [in=3072, out=4096] */
    struct ggml_tensor * wk;             /* bf16 [in=3072, out=1024] */
    struct ggml_tensor * wv;             /* bf16 [in=3072, out=1024] */
    struct ggml_tensor * wo;             /* bf16 [in=4096, out=3072] */
    struct ggml_tensor * attention_norm; /* f32  [3072] */

    /* Feed-forward */
    struct ggml_tensor * w1;             /* bf16 [in=3072, out=9216] */
    struct ggml_tensor * w2;             /* bf16 [in=9216, out=3072] */
    struct ggml_tensor * w3;             /* bf16 [in=3072, out=9216] */
    struct ggml_tensor * ffn_norm;       /* f32  [3072] */
} vox_dec_layer_w_t;

typedef struct {
    /* Tied: input embedding AND output projection */
    struct ggml_tensor * tok_embeddings; /* bf16 [in=3072, out=131072] */

    vox_dec_layer_w_t layers[VOX_DEC_LAYERS];

    struct ggml_tensor * norm;           /* f32  [3072] -- final decoder RMSNorm */
} vox_decoder_w_t;

typedef struct {
    vox_encoder_w_t encoder;
    vox_adapter_w_t adapter;
    vox_decoder_w_t decoder;

    /* Owned ggml resources -- vox_weights_free() releases these */
    struct ggml_context  * ctx;
    ggml_backend_buffer_t  buf;

    /* Stats (for logging) */
    int    n_tensors;
    size_t n_bytes;
} vox_weights_t;

/* Load all 711 Voxtral Realtime 4B weights from <model_dir>/consolidated.safetensors
 * into a single backend buffer of type `buft`. Returns NULL on any error.
 * Logs every missing/unmapped tensor to stderr. */
vox_weights_t * vox_weights_load(const char * model_dir, ggml_backend_buffer_type_t buft);

/* Free the backend buffer, ggml context, and the struct itself. */
void vox_weights_free(vox_weights_t * w);

#ifdef __cplusplus
}
#endif

#endif /* VOXTRAL_WEIGHTS_H */
