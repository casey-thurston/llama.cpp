/*
 * voxtral_stream.c -- vox_stream_t streaming inference library
 *
 * Sub-goal A: offline-pipeline placeholder. _feed buffers raw samples;
 * _finish runs the full pipeline (pad -> mel -> conv stem -> encoder graph
 * -> adapter graph -> decoder loop) and pushes generated token IDs to an
 * internal queue; _get drains decoded tokens to the caller.
 *
 * The pipeline body in run_offline_pipeline() is a strict refactor of the
 * code that previously lived inline in main.c -- no model logic changes.
 * Sub-goal C will replace it with a true incremental pipeline driven by
 * vox_stream_feed without changing voxtral_stream.h.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include "voxtral_stream.h"
#include "voxtral_tokenizer.h"
#include "voxtral_audio.h"
#include "voxtral_weights.h"
#include "voxtral_model.h"

/* voxtral_tokenizer.c declares this as extern; the library is the lowest
 * layer that always gets linked, so we own the definition here. */
int vox_verbose = 0;

/* ------------------------------------------------------------------ */
/* Streaming-style audio padding constants                            */
/* ------------------------------------------------------------------ */

#define RAW_AUDIO_LENGTH_PER_TOK   1280   /* 16000 / 12.5 Hz */
#define N_LEFT_PAD_TOKENS          32
#define N_RIGHT_PAD_TOKENS         17
#define TOKEN_BOS                  1
#define TOKEN_EOS                  2
#define TOKEN_STREAMING_PAD        32

#define VLOG(s, ...) do { if ((s)->opts.verbose) fprintf(stderr, __VA_ARGS__); } while (0)

/* ------------------------------------------------------------------ */
/* Internal: ada_state_t and kv_cache_t                               */
/* ------------------------------------------------------------------ */

typedef struct {
    struct ggml_context  * ctx;
    ggml_backend_buffer_t  buf;
    struct ggml_tensor   * scaled[26];
} ada_state_t;

typedef struct {
    struct ggml_context  * ctx;
    ggml_backend_buffer_t  buf;
    int                    max_seq;
    vox_kv_cache_layer_t   layers[26];
} kv_cache_t;

typedef struct {
    struct ggml_context  * ctx;
    ggml_backend_buffer_t  buf;
    int                    max_seq;
    vox_enc_kv_layer_t     layers[VOX_ENC_LAYERS_HDR];
} enc_kv_cache_t;

/* ------------------------------------------------------------------ */
/* Public opaque struct                                               */
/* ------------------------------------------------------------------ */

struct vox_stream {
    vox_stream_opts_t          opts;

    vox_tokenizer_t          * tok;
    ggml_backend_t             backend;
    ggml_backend_buffer_type_t buft;
    vox_weights_t            * w;

    ada_state_t                ada;
    kv_cache_t                 dec_kv;
    enc_kv_cache_t             enc_kv;
    int                        max_n_audio;
    int                        max_n_enc;

    /* Cached F32 conv stem weights, loaded once at init. ~21 MB total. */
    float * conv0_w;     /* [VOX_ENC_DIM, VOX_MEL_BINS, 3] */
    float * conv0_b;     /* [VOX_ENC_DIM] */
    float * conv1_w;     /* [VOX_ENC_DIM, VOX_ENC_DIM, 3] */
    float * conv1_b;     /* [VOX_ENC_DIM] */

    /* Reusable per-step buffers. */
    float * tok_embed_buf;    /* [VOX_DEC_DIM] */
    float * input_embed_buf;  /* [VOX_DEC_DIM] */
    float * logits_buf;       /* [VOX_VOCAB_SIZE] */

    /* Streaming pipeline state (reset by vox_stream_reset). */
    vox_mel_ctx_t * mel_ctx;
    int   total_samples_fed;
    int   mel_consumed;
    int   c0_produced;
    int   c1_produced;
    int   n_enc_total;
    int   n_audio_total;
    int   decoder_pos;
    int   prev_token;
    int   eos_seen;
    int   finished;

    float mel_tail[VOX_MEL_BINS * 2];   /* [128, 2] channel-major */
    int   mel_tail_count;
    float c0_tail[VOX_ENC_DIM * 2];     /* [1280, 2] channel-major */
    int   c0_tail_count;

    float * enc_outputs;        /* [max_n_enc * VOX_ENC_DIM] */
    float * audio_embeds;       /* [max_n_audio * VOX_DEC_DIM] */

    int * tokens;
    int   tokens_cap;
    int   tokens_len;
    int   tokens_emitted;

    /* Legacy sample buffer for the offline placeholder path. */
    float * samples;
    int     samples_cap;
    int     samples_len;
};

/* ------------------------------------------------------------------ */
/* Helpers (lifted verbatim from main.c)                              */
/* ------------------------------------------------------------------ */

/* CPU causal conv1d -- matches voxtral.c's vox_causal_conv1d, with the BLAS
 * dependency replaced by manual loops. */
static float * cpu_causal_conv1d(
    const float * in,        /* [channels_in, length] row-major */
    const float * weight,    /* [channels_out, channels_in, kernel] row-major */
    const float * bias,      /* [channels_out] (may be NULL) */
    int channels_in, int channels_out, int length,
    int kernel_size, int stride,
    int * out_length_ptr)
{
    int padding_total = kernel_size - stride;
    int out_length = (int) ceilf(((float) length - kernel_size + padding_total) / (float) stride + 1.0f);
    if (out_length <= 0) { *out_length_ptr = 0; return NULL; }
    int left_pad = padding_total;

    float * out = (float *) calloc((size_t) channels_out * out_length, sizeof(float));
    if (!out) return NULL;

    for (int oc = 0; oc < channels_out; oc++) {
        const float * w_row = weight + (size_t) oc * channels_in * kernel_size;
        float b = bias ? bias[oc] : 0.0f;
        for (int ol = 0; ol < out_length; ol++) {
            float acc = b;
            for (int ic = 0; ic < channels_in; ic++) {
                int weight_base = ic * kernel_size;
                size_t in_row = (size_t) ic * length;
                for (int k = 0; k < kernel_size; k++) {
                    int il = ol * stride - left_pad + k;
                    if (il < 0 || il >= length) continue;
                    acc += w_row[weight_base + k] * in[in_row + il];
                }
            }
            out[(size_t) oc * out_length + ol] = acc;
        }
    }

    *out_length_ptr = out_length;
    return out;
}

/* GELU (matches voxtral.c's vox_gelu and PyTorch's tanh approximation). */
static void cpu_gelu_inplace(float * x, int n) {
    const float c1 = 0.7978845608f;        /* sqrt(2/pi) */
    const float c2 = 0.044715f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float t = c1 * (v + c2 * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(t));
    }
}

/* Sinusoidal time embedding -- matches voxtral.c's vox_compute_time_embedding
 * and the Mistral vLLM reference. Output layout: [cos(...) | sin(...)]. */
static void cpu_compute_time_embedding(float * out, float t_value, int dim) {
    int half = dim / 2;
    float log_theta = logf(10000.0f);
    for (int i = 0; i < half; i++) {
        float inv_freq = expf(-log_theta * (float) i / (float) half);
        float emb = t_value * inv_freq;
        out[i]        = cosf(emb);
        out[i + half] = sinf(emb);
    }
}

static void read_f32_tensor(const struct ggml_tensor * t, float * dst) {
    ggml_backend_tensor_get(t, dst, 0, ggml_nbytes(t));
}

/* Read one row out of a 2D tensor (BF16 or quantized) and produce F32.
 * Used for the tied tok_embeddings lookup in the decoder loop. */
static void read_row_as_f32(const struct ggml_tensor * t, int row_idx, float * dst) {
    int64_t embed_dim = t->ne[0];
    size_t row_bytes = t->nb[1];  /* stride between rows in bytes */

    if (t->type == GGML_TYPE_BF16) {
        size_t bf16_bytes = (size_t) embed_dim * sizeof(uint16_t);
        uint16_t * tmp = (uint16_t *) malloc(bf16_bytes);
        ggml_backend_tensor_get(t, tmp, (size_t) row_idx * row_bytes, bf16_bytes);
        for (int64_t i = 0; i < embed_dim; i++) {
            uint32_t bits = ((uint32_t) tmp[i]) << 16;
            float f;
            memcpy(&f, &bits, sizeof(f));
            dst[i] = f;
        }
        free(tmp);
    } else {
        /* Quantized type: read raw row bytes, dequantize via type traits. */
        void * tmp = malloc(row_bytes);
        ggml_backend_tensor_get(t, tmp, (size_t) row_idx * row_bytes, row_bytes);
        const struct ggml_type_traits * tt = ggml_get_type_traits(t->type);
        tt->to_float(tmp, dst, embed_dim);
        free(tmp);
    }
}

/* Pad audio streaming-style: prepend N_LEFT_PAD_TOKENS*1280 zeros, append
 * align-to-1280 zeros plus N_RIGHT_PAD_TOKENS*1280 zeros. Caller frees. */
static float * pad_audio_streaming(const float * in, int n_in, int * n_out_ptr) {
    int align_pad = (RAW_AUDIO_LENGTH_PER_TOK - (n_in % RAW_AUDIO_LENGTH_PER_TOK)) % RAW_AUDIO_LENGTH_PER_TOK;
    int right_pad = align_pad + N_RIGHT_PAD_TOKENS * RAW_AUDIO_LENGTH_PER_TOK;
    int left_pad  = N_LEFT_PAD_TOKENS * RAW_AUDIO_LENGTH_PER_TOK;
    int n_out = left_pad + n_in + right_pad;
    float * out = (float *) calloc(n_out, sizeof(float));
    if (!out) return NULL;
    memcpy(out + left_pad, in, (size_t) n_in * sizeof(float));
    *n_out_ptr = n_out;
    return out;
}

/* Run the conv stem in plain C and produce the [VOX_ENC_DIM, conv_seq_len]
 * F32 input that the encoder graph expects. See main.c history for details. */
static float * run_conv_stem(
    const vox_weights_t * w,
    const float * mel_t_first,   /* [n_mel_frames, 128] row-major */
    int n_mel_frames,
    int * out_seq_len_ptr)
{
    int conv0_w_n = (int) ggml_nelements(w->encoder.conv0);
    int conv1_w_n = (int) ggml_nelements(w->encoder.conv1);
    float * conv0_w = (float *) malloc(conv0_w_n * sizeof(float));
    float * conv1_w = (float *) malloc(conv1_w_n * sizeof(float));
    float * conv0_b = (float *) malloc(VOX_ENC_DIM * sizeof(float));
    float * conv1_b = (float *) malloc(VOX_ENC_DIM * sizeof(float));
    read_f32_tensor(w->encoder.conv0,      conv0_w);
    read_f32_tensor(w->encoder.conv1,      conv1_w);
    read_f32_tensor(w->encoder.conv0_bias, conv0_b);
    read_f32_tensor(w->encoder.conv1_bias, conv1_b);

    /* Transpose mel from [n_mel_frames, 128] to [128, n_mel_frames]. */
    float * conv_in = (float *) malloc((size_t) VOX_MEL_BINS * n_mel_frames * sizeof(float));
    for (int f = 0; f < n_mel_frames; f++) {
        for (int m = 0; m < VOX_MEL_BINS; m++) {
            conv_in[(size_t) m * n_mel_frames + f] = mel_t_first[(size_t) f * VOX_MEL_BINS + m];
        }
    }

    int conv0_out = 0;
    float * after_conv0 = cpu_causal_conv1d(
        conv_in, conv0_w, conv0_b,
        VOX_MEL_BINS, VOX_ENC_DIM, n_mel_frames,
        /*kernel*/ 3, /*stride*/ 1, &conv0_out);
    cpu_gelu_inplace(after_conv0, VOX_ENC_DIM * conv0_out);
    free(conv_in);
    free(conv0_w);
    free(conv0_b);

    int conv1_out = 0;
    float * after_conv1 = cpu_causal_conv1d(
        after_conv0, conv1_w, conv1_b,
        VOX_ENC_DIM, VOX_ENC_DIM, conv0_out,
        /*kernel*/ 3, /*stride*/ 2, &conv1_out);
    cpu_gelu_inplace(after_conv1, VOX_ENC_DIM * conv1_out);
    free(after_conv0);
    free(conv1_w);
    free(conv1_b);

    /* Transpose [channels_out, out_length] -> ggml [enc_dim, n_pos] order
     * (n_pos becomes the slow axis). */
    float * out = (float *) malloc((size_t) VOX_ENC_DIM * conv1_out * sizeof(float));
    for (int p = 0; p < conv1_out; p++) {
        for (int c = 0; c < VOX_ENC_DIM; c++) {
            out[(size_t) p * VOX_ENC_DIM + c] = after_conv1[(size_t) c * conv1_out + p];
        }
    }
    free(after_conv1);

    *out_seq_len_ptr = conv1_out;
    return out;
}

/* Sliding-window causal mask: element (i, j) is 0 if reachable, else -INF. */
static void build_encoder_mask(float * mask, int n_pos) {
    const int window = VOX_ENC_WINDOW;
    for (int i = 0; i < n_pos; i++) {
        for (int j = 0; j < n_pos; j++) {
            int allowed = (j <= i) && (j >= i - window + 1);
            mask[(size_t) i * n_pos + j] = allowed ? 0.0f : -INFINITY;
        }
    }
}

/* Encoder step mask: rows = new query positions (absolute kv_pos..kv_pos+n_new-1),
 * cols = cache positions [0..n_total). 0 if reachable per causal sliding window,
 * else -INF. Layout: row-major, ne[0]=n_total (col), ne[1]=n_new (row). */
static void build_encoder_step_mask(float * mask, int n_total, int n_new, int kv_pos) {
    const int window = VOX_ENC_WINDOW;
    for (int r = 0; r < n_new; r++) {
        const int abs_q = kv_pos + r;
        for (int c = 0; c < n_total; c++) {
            int allowed = (c <= abs_q) && (c >= abs_q - window + 1);
            mask[(size_t) r * n_total + c] = allowed ? 0.0f : -INFINITY;
        }
    }
}

/* ------------------------------------------------------------------ */
/* ada_state                                                          */
/* ------------------------------------------------------------------ */

static int ada_state_compute(ada_state_t * a, const vox_weights_t * w,
                              ggml_backend_buffer_type_t buft, int delay_tokens) {
    struct ggml_init_params ip = {
        /* .mem_size  = */ ggml_tensor_overhead() * 64,
        /* .mem_buffer = */ NULL,
        /* .no_alloc  = */ true,
    };
    a->ctx = ggml_init(ip);
    if (!a->ctx) return -1;

    char name[32];
    for (int i = 0; i < 26; i++) {
        a->scaled[i] = ggml_new_tensor_1d(a->ctx, GGML_TYPE_F32, VOX_DEC_DIM);
        snprintf(name, sizeof(name), "ada_scaled.%d", i);
        ggml_set_name(a->scaled[i], name);
    }
    a->buf = ggml_backend_alloc_ctx_tensors_from_buft(a->ctx, buft);
    if (!a->buf) return -1;

    float t_cond[VOX_DEC_DIM];
    cpu_compute_time_embedding(t_cond, (float) delay_tokens, VOX_DEC_DIM);

    float * down_w = (float *) malloc(VOX_DEC_DIM * 32 * sizeof(float));
    float * up_w   = (float *) malloc(32 * VOX_DEC_DIM * sizeof(float));
    float hidden[32];
    float scaled[VOX_DEC_DIM];

    for (int i = 0; i < 26; i++) {
        const vox_dec_layer_w_t * lw = &w->decoder.layers[i];

        read_f32_tensor(lw->ada_norm_down, down_w);
        for (int oi = 0; oi < 32; oi++) {
            float acc = 0.0f;
            const float * row = down_w + (size_t) oi * VOX_DEC_DIM;
            for (int j = 0; j < VOX_DEC_DIM; j++) acc += row[j] * t_cond[j];
            hidden[oi] = acc;
        }
        cpu_gelu_inplace(hidden, 32);

        read_f32_tensor(lw->ada_norm_up, up_w);
        for (int oi = 0; oi < VOX_DEC_DIM; oi++) {
            float acc = 0.0f;
            const float * row = up_w + (size_t) oi * 32;
            for (int j = 0; j < 32; j++) acc += row[j] * hidden[j];
            scaled[oi] = 1.0f + acc;
        }

        ggml_backend_tensor_set(a->scaled[i], scaled, 0, sizeof(scaled));
    }

    free(down_w);
    free(up_w);
    return 0;
}

static void ada_state_free(ada_state_t * a) {
    if (a->buf) ggml_backend_buffer_free(a->buf);
    if (a->ctx) ggml_free(a->ctx);
    memset(a, 0, sizeof(*a));
}

/* ------------------------------------------------------------------ */
/* KV cache                                                           */
/* ------------------------------------------------------------------ */

static int kv_cache_alloc(kv_cache_t * c, ggml_backend_buffer_type_t buft, int max_seq) {
    struct ggml_init_params ip = {
        /* .mem_size  = */ ggml_tensor_overhead() * 64,
        /* .mem_buffer = */ NULL,
        /* .no_alloc  = */ true,
    };
    c->ctx = ggml_init(ip);
    if (!c->ctx) return -1;
    c->max_seq = max_seq;
    const int kv_dim = VOX_DEC_HEAD_DIM * VOX_DEC_KV_HEADS;
    char name[32];
    for (int i = 0; i < 26; i++) {
        c->layers[i].k = ggml_new_tensor_2d(c->ctx, GGML_TYPE_F32, kv_dim, max_seq);
        snprintf(name, sizeof(name), "kv.k.%d", i);
        ggml_set_name(c->layers[i].k, name);
        c->layers[i].v = ggml_new_tensor_2d(c->ctx, GGML_TYPE_F32, kv_dim, max_seq);
        snprintf(name, sizeof(name), "kv.v.%d", i);
        ggml_set_name(c->layers[i].v, name);
    }
    c->buf = ggml_backend_alloc_ctx_tensors_from_buft(c->ctx, buft);
    if (!c->buf) return -1;
    return 0;
}

static void kv_cache_free(kv_cache_t * c) {
    if (c->buf) ggml_backend_buffer_free(c->buf);
    if (c->ctx) ggml_free(c->ctx);
    memset(c, 0, sizeof(*c));
}

static int enc_kv_cache_alloc(enc_kv_cache_t * c, ggml_backend_buffer_type_t buft, int max_seq) {
    struct ggml_init_params ip = {
        /* .mem_size  = */ ggml_tensor_overhead() * 128,
        /* .mem_buffer = */ NULL,
        /* .no_alloc  = */ true,
    };
    c->ctx = ggml_init(ip);
    if (!c->ctx) return -1;
    c->max_seq = max_seq;
    const int kv_dim = VOX_ENC_HEAD_DIM * VOX_ENC_HEADS;  /* full MHA */
    char name[32];
    for (int i = 0; i < VOX_ENC_LAYERS_HDR; i++) {
        c->layers[i].k = ggml_new_tensor_2d(c->ctx, GGML_TYPE_F32, kv_dim, max_seq);
        snprintf(name, sizeof(name), "enc_kv.k.%d", i);
        ggml_set_name(c->layers[i].k, name);
        c->layers[i].v = ggml_new_tensor_2d(c->ctx, GGML_TYPE_F32, kv_dim, max_seq);
        snprintf(name, sizeof(name), "enc_kv.v.%d", i);
        ggml_set_name(c->layers[i].v, name);
    }
    c->buf = ggml_backend_alloc_ctx_tensors_from_buft(c->ctx, buft);
    if (!c->buf) return -1;
    return 0;
}

static void enc_kv_cache_free(enc_kv_cache_t * c) {
    if (c->buf) ggml_backend_buffer_free(c->buf);
    if (c->ctx) ggml_free(c->ctx);
    memset(c, 0, sizeof(*c));
}

/* ------------------------------------------------------------------ */
/* Token queue                                                        */
/* ------------------------------------------------------------------ */

static int tokens_push(vox_stream_t * s, int id) {
    if (s->tokens_len == s->tokens_cap) {
        int new_cap = s->tokens_cap > 0 ? s->tokens_cap * 2 : 256;
        int * grown = (int *) realloc(s->tokens, (size_t) new_cap * sizeof(int));
        if (!grown) return -1;
        s->tokens = grown;
        s->tokens_cap = new_cap;
    }
    s->tokens[s->tokens_len++] = id;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Streaming conv1d                                                    */
/* ------------------------------------------------------------------ */

/* Streaming-friendly conv1d. Computes `global_out_count` output positions
 * starting at global output position `global_out_start`. Input data comes
 * from `in_buf` which holds `in_count` frames starting at global input
 * position `global_in_start`. Layout: [channels_in, in_count] row-major
 * (channel-major). Returns a freshly allocated [channels_out, global_out_count]
 * buffer in the same layout.
 *
 * For input positions gi < 0 (initial left-padding zone), the value is treated
 * as zero. For gi outside [global_in_start, global_in_start + in_count), the
 * caller must ensure this doesn't happen (the buffer must cover all needed
 * input positions except the initial left-pad zone). */
static float * cpu_streaming_conv1d(
    const float * in_buf, int in_count, int global_in_start,
    const float * weight, const float * bias,
    int channels_in, int channels_out,
    int kernel, int stride, int left_pad,
    int global_out_start, int global_out_count)
{
    if (global_out_count <= 0) return NULL;
    float * out = (float *) calloc((size_t) channels_out * global_out_count, sizeof(float));
    if (!out) return NULL;

    for (int oc = 0; oc < channels_out; oc++) {
        const float * w_row = weight + (size_t) oc * channels_in * kernel;
        float b = bias ? bias[oc] : 0.0f;
        for (int og = 0; og < global_out_count; og++) {
            int p = global_out_start + og;
            float acc = b;
            for (int ic = 0; ic < channels_in; ic++) {
                int wb = ic * kernel;
                for (int k = 0; k < kernel; k++) {
                    int gi = p * stride - left_pad + k;
                    if (gi < 0) continue;
                    int li = gi - global_in_start;
                    if (li < 0 || li >= in_count) continue;
                    acc += w_row[wb + k] * in_buf[(size_t) ic * in_count + li];
                }
            }
            out[(size_t) oc * global_out_count + og] = acc;
        }
    }
    return out;
}

/* ------------------------------------------------------------------ */
/* Streaming pipeline helpers                                          */
/* ------------------------------------------------------------------ */

/* Copy the last `want` frames from a [channels, total_count] buffer into
 * `dst` (also [channels, want]). If total_count < want, copies all of it. */
static void copy_tail_frames(float * dst, const float * src,
                             int channels, int total_count, int want) {
    int have = total_count < want ? total_count : want;
    int skip = total_count - have;
    for (int c = 0; c < channels; c++) {
        memcpy(dst + (size_t) c * have,
               src + (size_t) c * total_count + skip,
               (size_t) have * sizeof(float));
    }
}

/* Transpose [n_frames, channels] row-major → [channels, n_frames] into dst. */
static void transpose_mel_to_channel_major(float * dst, const float * src,
                                           int n_frames, int channels) {
    for (int f = 0; f < n_frames; f++) {
        for (int c = 0; c < channels; c++) {
            dst[(size_t) c * n_frames + f] = src[(size_t) f * channels + c];
        }
    }
}

/* Run the conv stem as a ggml graph on the CUDA backend. The conv1d weights
 * are already on GPU (loaded by the weight loader). We left-pad the input
 * manually on CPU (since ggml_conv_1d uses symmetric padding, not causal
 * left-only), build a small graph, run it on GPU, and read back the result.
 *
 * For chunk-boundary streaming: mel_tail provides left context for conv0,
 * c0_tail provides left context for conv1. The graph recomputes the tail
 * positions redundantly (cheap on GPU) and we slice out only the new ones. */
static float * run_conv_stem_stream(vox_stream_t * s, int * out_count) {
    *out_count = 0;

    /* Get new mel frames from the incremental mel context. */
    int n_mel_in_ctx = 0;
    float * mel_buf = vox_mel_data(s->mel_ctx, &n_mel_in_ctx);
    int mel_offset = vox_mel_frame_offset(s->mel_ctx);
    int n_new_mel = (mel_offset + n_mel_in_ctx) - s->mel_consumed;
    if (n_new_mel <= 0) return NULL;

    /* ---- Prepare mel input with left context ---- */
    int n_in_mel = s->mel_tail_count + n_new_mel;

    /* Build mel input as [n_in_mel, 128] ggml-layout (ne[0]=128, ne[1]=n_in_mel).
     * mel_tail is already in this layout. New mel from mel_ctx is [n, 128]
     * row-major which is the SAME ggml layout (ne[0]=128, ne[1]=n). */
    float * mel_in_ggml = (float *) malloc((size_t) VOX_MEL_BINS * n_in_mel * sizeof(float));
    if (!mel_in_ggml) return NULL;
    if (s->mel_tail_count > 0)
        memcpy(mel_in_ggml, s->mel_tail, (size_t) VOX_MEL_BINS * s->mel_tail_count * sizeof(float));
    int new_mel_local = s->mel_consumed - mel_offset;
    memcpy(mel_in_ggml + (size_t) s->mel_tail_count * VOX_MEL_BINS,
           mel_buf + (size_t) new_mel_local * VOX_MEL_BINS,
           (size_t) n_new_mel * VOX_MEL_BINS * sizeof(float));

    /* Left-pad with 2 zero frames for conv0's causal padding. */
    int mel_padded_len = 2 + n_in_mel;
    float * mel_padded = (float *) calloc((size_t) VOX_MEL_BINS * mel_padded_len, sizeof(float));
    if (!mel_padded) { free(mel_in_ggml); return NULL; }
    memcpy(mel_padded + 2 * VOX_MEL_BINS, mel_in_ggml, (size_t) VOX_MEL_BINS * n_in_mel * sizeof(float));

    /* Update mel_tail: keep last 2 frames of mel_in_ggml. */
    int mt_want = n_in_mel < 2 ? n_in_mel : 2;
    memcpy(s->mel_tail, mel_in_ggml + (size_t) (n_in_mel - mt_want) * VOX_MEL_BINS,
           (size_t) mt_want * VOX_MEL_BINS * sizeof(float));
    s->mel_tail_count = mt_want;
    free(mel_in_ggml);

    /* conv0 output length: (mel_padded_len - 3) / 1 + 1 = mel_padded_len - 2 = n_in_mel. */
    int c0_full_len = n_in_mel;
    int c0_new_count = n_new_mel;

    /* c0_tail bookkeeping: c0_full includes tail positions at the front.
     * The c0_tail_count positions are stale; the last c0_new_count are new. */
    int new_c0_produced = s->c0_produced + c0_new_count;

    /* c1 output count: how many new c1 positions can we produce? */
    int p_max = (new_c0_produced - 2) / 2;
    int n_new_c1 = p_max + 1 - s->c1_produced;

    /* ---- Build the ggml graph ----
     * ggml_conv_1d expects kernel [K, IC, OC] and data [L, IC] where ne[0]=L.
     * Our mel tensor is [mel_padded_len, 128] but in ggml ne[0]=128. We need
     * ne[0]=mel_padded_len, so we transpose the data for ggml_conv_1d. */

    struct ggml_init_params ip = {
        ggml_tensor_overhead() * 64 + ggml_graph_overhead(),
        NULL, true,
    };
    struct ggml_context * gctx = ggml_init(ip);
    if (!gctx) { free(mel_padded); return NULL; }

    /* Input: transposed mel [L=mel_padded_len, IC=128]. */
    struct ggml_tensor * mel_t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, mel_padded_len, VOX_MEL_BINS);
    ggml_set_name(mel_t, "conv.mel"); ggml_set_input(mel_t);

    /* Conv0: p=0 (causal pad already in input), s=1, d=1. */
    struct ggml_tensor * c0_raw = ggml_conv_1d(gctx, s->w->encoder.conv0, mel_t, 1, 0, 1);
    /* c0_raw shape: [c0_full_len, 1280, 1]. Permute to [1280, c0_full_len] for bias. */
    struct ggml_tensor * c0_t = ggml_permute(gctx, c0_raw, 1, 0, 2, 3);
    struct ggml_tensor * c0 = ggml_add(gctx, c0_t, s->w->encoder.conv0_bias);
    c0 = ggml_gelu(gctx, c0);

    /* Prepare for conv1: need data as [L, IC=1280]. c0 is [1280, c0_full_len].
     * Permute back to [c0_full_len, 1280] and make contiguous. */
    struct ggml_tensor * c0_for_c1 = ggml_cont(gctx, ggml_permute(gctx, c0, 1, 0, 2, 3));

    /* Left-pad c0 with 1 zero column for conv1's causal padding. We use
     * ggml_pad which pads the ne[0] dimension. c0_for_c1 has ne[0]=c0_full_len.
     * ggml_pad pads on the RIGHT; we need LEFT. So we'll pad the input and
     * accept a slightly different output count.
     *
     * Actually: ggml_pad pads with zeros at the end of each dimension. For LEFT
     * padding, the simplest approach is to create a zero tensor and concat. */
    struct ggml_tensor * zero_col = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 1, VOX_ENC_DIM);
    ggml_set_name(zero_col, "conv.zero"); ggml_set_input(zero_col);
    struct ggml_tensor * c0_padded = ggml_concat(gctx, zero_col, c0_for_c1, 0);

    /* Conv1: stride=2, p=0. */
    struct ggml_tensor * c1_raw = ggml_conv_1d(gctx, s->w->encoder.conv1, c0_padded, 2, 0, 1);
    struct ggml_tensor * c1_t = ggml_permute(gctx, c1_raw, 1, 0, 2, 3);
    struct ggml_tensor * c1 = ggml_add(gctx, c1_t, s->w->encoder.conv1_bias);
    c1 = ggml_gelu(gctx, c1);
    ggml_set_name(c1, "conv.output"); ggml_set_output(c1);

    struct ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, c1);

    ggml_gallocr_t galloc = ggml_gallocr_new(s->buft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc); ggml_free(gctx); free(mel_padded);
        fprintf(stderr, "vox_stream: conv stem gallocr failed\n");
        return NULL;
    }

    /* Set mel input: transpose [mel_padded_len, 128] ggml-layout (ne[0]=128)
     * to [mel_padded_len, 128] with ne[0]=mel_padded_len (channel-major). */
    float * mel_cm = (float *) malloc((size_t) mel_padded_len * VOX_MEL_BINS * sizeof(float));
    if (!mel_cm) { ggml_gallocr_free(galloc); ggml_free(gctx); free(mel_padded); return NULL; }
    for (int f = 0; f < mel_padded_len; f++)
        for (int c = 0; c < VOX_MEL_BINS; c++)
            mel_cm[(size_t) c * mel_padded_len + f] = mel_padded[(size_t) f * VOX_MEL_BINS + c];
    ggml_backend_tensor_set(mel_t, mel_cm, 0, (size_t) mel_padded_len * VOX_MEL_BINS * sizeof(float));
    free(mel_cm);
    free(mel_padded);

    /* Set zero column (already zeroed by calloc in tensor alloc? No — backend
     * buffer is uninitialized. Set explicitly.) */
    float zero_buf[VOX_ENC_DIM];
    memset(zero_buf, 0, sizeof(zero_buf));
    ggml_backend_tensor_set(zero_col, zero_buf, 0, sizeof(zero_buf));

    if (ggml_backend_graph_compute(s->backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc); ggml_free(gctx);
        fprintf(stderr, "vox_stream: conv stem graph_compute failed\n");
        return NULL;
    }

    /* Read output: c1 has shape [1280, c1_full_len].
     * c1_full_len = (c0_full_len + 1 - 3) / 2 + 1 = (n_in_mel - 1) / 2. */
    int64_t c1_full_len = c1->ne[1];

    /* Slice out only the NEW c1 positions. The output contains positions for
     * both the tail-context zone and the new zone. We need the last n_new_c1. */
    if (n_new_c1 <= 0 || n_new_c1 > (int) c1_full_len) {
        /* c0_tail update for when conv1 can't produce anything yet. */
        /* Read c0 output to get new tail frames. */
        /* For now, fall back to accepting whatever c1 we got. */
        if (n_new_c1 <= 0) {
            ggml_gallocr_free(galloc); ggml_free(gctx);
            s->mel_consumed += n_new_mel;
            s->c0_produced  += c0_new_count;
            /* Update c0_tail from the conv0 GPU output. Read last 2 frames. */
            /* TODO: this path needs c0 readback. For now it's rare (< 4 mel frames). */
            return NULL;
        }
    }

    int c1_skip = (int) c1_full_len - n_new_c1;
    float * enc_in = (float *) malloc((size_t) VOX_ENC_DIM * n_new_c1 * sizeof(float));
    if (enc_in) {
        /* c1 has ggml ne[0]=1280, ne[1]=c1_full_len. Memory: [c1_full_len, 1280]
         * with ne[0]=1280 innermost. Read starting at position c1_skip. */
        ggml_backend_tensor_get(c1, enc_in,
            (size_t) c1_skip * VOX_ENC_DIM * sizeof(float),
            (size_t) n_new_c1 * VOX_ENC_DIM * sizeof(float));
    }

    ggml_gallocr_free(galloc);
    ggml_free(gctx);

    s->mel_consumed += n_new_mel;
    s->c0_produced  += c0_new_count;
    s->c1_produced  += n_new_c1;

    *out_count = n_new_c1;
    return enc_in;
}

/* Run the encoder step graph for n_new new positions. Reads enc_in as
 * [VOX_ENC_DIM, n_new] (ggml layout). Writes output hidden states to
 * s->enc_outputs at positions [n_enc_total, n_enc_total + n_new). */
static int run_encoder_step(vox_stream_t * s, const float * enc_in, int n_new) {
    if (n_new <= 0) return 0;
    int kv_pos = s->n_enc_total;
    int n_total = s->n_enc_total + n_new;

    struct ggml_init_params ip = {
        ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false),
        NULL, true,
    };
    struct ggml_context * gctx = ggml_init(ip);

    vox_encoder_step_graph_t eg = vox_build_encoder_step_graph(
        gctx, s->w, s->enc_kv.layers, n_new, n_total, kv_pos);

    ggml_gallocr_t galloc = ggml_gallocr_new(s->buft);
    if (!ggml_gallocr_alloc_graph(galloc, eg.gf)) {
        ggml_gallocr_free(galloc); ggml_free(gctx);
        fprintf(stderr, "vox_stream: encoder step gallocr failed\n");
        return -1;
    }

    ggml_backend_tensor_set(eg.input, enc_in, 0, (size_t) VOX_ENC_DIM * n_new * sizeof(float));

    int32_t * pos_buf = (int32_t *) malloc(n_new * sizeof(int32_t));
    for (int i = 0; i < n_new; i++) pos_buf[i] = kv_pos + i;
    ggml_backend_tensor_set(eg.pos, pos_buf, 0, n_new * sizeof(int32_t));
    free(pos_buf);

    float * mask_buf = (float *) malloc((size_t) n_total * n_new * sizeof(float));
    build_encoder_step_mask(mask_buf, n_total, n_new, kv_pos);
    ggml_backend_tensor_set(eg.mask, mask_buf, 0, (size_t) n_total * n_new * sizeof(float));
    free(mask_buf);

    if (ggml_backend_graph_compute(s->backend, eg.gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc); ggml_free(gctx);
        fprintf(stderr, "vox_stream: encoder step graph_compute failed\n");
        return -1;
    }

    ggml_backend_tensor_get(eg.output,
        s->enc_outputs + (size_t) kv_pos * VOX_ENC_DIM,
        0, (size_t) VOX_ENC_DIM * n_new * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(gctx);

    s->n_enc_total = n_total;
    return 0;
}

/* Run the adapter for 4 contiguous encoder positions at enc_start.
 * Writes 1 audio embedding to s->audio_embeds[n_audio_total]. */
static int run_adapter_step(vox_stream_t * s, int enc_start) {
    struct ggml_init_params ip = {
        ggml_tensor_overhead() * 64 + ggml_graph_overhead(),
        NULL, true,
    };
    struct ggml_context * gctx = ggml_init(ip);
    vox_adapter_graph_t ag = vox_build_adapter_graph(gctx, s->w, VOX_DOWNSAMPLE);

    ggml_gallocr_t galloc = ggml_gallocr_new(s->buft);
    if (!ggml_gallocr_alloc_graph(galloc, ag.gf)) {
        ggml_gallocr_free(galloc); ggml_free(gctx);
        return -1;
    }

    ggml_backend_tensor_set(ag.input,
        s->enc_outputs + (size_t) enc_start * VOX_ENC_DIM,
        0, (size_t) VOX_ENC_DIM * VOX_DOWNSAMPLE * sizeof(float));

    if (ggml_backend_graph_compute(s->backend, ag.gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc); ggml_free(gctx);
        return -1;
    }

    ggml_backend_tensor_get(ag.output,
        s->audio_embeds + (size_t) s->n_audio_total * VOX_DEC_DIM,
        0, VOX_DEC_DIM * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(gctx);

    s->n_audio_total++;
    return 0;
}

/* Run a single decoder step at position `pos`. Writes the argmax token to
 * the token queue if past the prompt and not EOS. */
static int run_decoder_step_streaming(vox_stream_t * s) {
    int pos = s->decoder_pos;
    int prompt_len = 1 + N_LEFT_PAD_TOKENS + s->opts.delay_tokens;
    int prompt[64];
    prompt[0] = TOKEN_BOS;
    for (int i = 1; i < prompt_len; i++) prompt[i] = TOKEN_STREAMING_PAD;

    int text_token = (pos < prompt_len) ? prompt[pos] : s->prev_token;

    read_row_as_f32(s->w->decoder.tok_embeddings, text_token, s->tok_embed_buf);
    const float * audio = s->audio_embeds + (size_t) pos * VOX_DEC_DIM;
    for (int i = 0; i < VOX_DEC_DIM; i++)
        s->input_embed_buf[i] = audio[i] + s->tok_embed_buf[i];

    const int n_kv = pos + 1;
    const int kv_pos = pos;
    struct ggml_init_params ip = {
        ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false),
        NULL, true,
    };
    struct ggml_context * gctx = ggml_init(ip);

    struct ggml_tensor * ada_arr[26];
    for (int i = 0; i < 26; i++) ada_arr[i] = s->ada.scaled[i];

    vox_decoder_graph_t dg = vox_build_decoder_graph(gctx, s->w, s->dec_kv.layers,
                                                      ada_arr, n_kv, kv_pos, 1);
    ggml_gallocr_t galloc = ggml_gallocr_new(s->buft);
    if (!ggml_gallocr_alloc_graph(galloc, dg.gf)) {
        ggml_gallocr_free(galloc); ggml_free(gctx);
        return -1;
    }

    ggml_backend_tensor_set(dg.input, s->input_embed_buf, 0, VOX_DEC_DIM * sizeof(float));
    int32_t pos_val = pos;
    ggml_backend_tensor_set(dg.pos, &pos_val, 0, sizeof(int32_t));

    float * mask_buf = (float *) calloc(n_kv, sizeof(float));
    ggml_backend_tensor_set(dg.mask, mask_buf, 0, n_kv * sizeof(float));
    free(mask_buf);

    if (ggml_backend_graph_compute(s->backend, dg.gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc); ggml_free(gctx);
        return -1;
    }

    ggml_backend_tensor_get(dg.logits, s->logits_buf, 0, VOX_VOCAB_SIZE * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(gctx);

    int argmax = 0;
    float lmax = s->logits_buf[0];
    for (int i = 1; i < VOX_VOCAB_SIZE; i++) {
        if (s->logits_buf[i] > lmax) { lmax = s->logits_buf[i]; argmax = i; }
    }

    if (pos >= prompt_len - 1) {
        if (argmax == TOKEN_EOS) {
            VLOG(s, "  pos %d: EOS\n", pos);
            s->eos_seen = 1;
        } else {
            tokens_push(s, argmax);
            s->prev_token = argmax;
        }
    }

    s->decoder_pos++;
    return 0;
}

/* Prefill prompt_len decoder positions in a single multi-token forward pass.
 * The prompt is [BOS, STREAMING_PAD*38]; generation starts at the LAST prompt
 * position (index prompt_len-1). Produces at most 1 token (the first generated
 * token from the last prompt position's logits). */
static int run_decoder_prefill(vox_stream_t * s, int prompt_len) {
    /* Build the combined input: audio_embed[pos] + tok_embed(prompt[pos])
     * for pos in [0, prompt_len). */
    float * input_buf = (float *) malloc((size_t) VOX_DEC_DIM * prompt_len * sizeof(float));
    if (!input_buf) return -1;

    int prompt[64];
    prompt[0] = TOKEN_BOS;
    for (int i = 1; i < prompt_len; i++) prompt[i] = TOKEN_STREAMING_PAD;

    for (int pos = 0; pos < prompt_len; pos++) {
        read_row_as_f32(s->w->decoder.tok_embeddings, prompt[pos], s->tok_embed_buf);
        const float * audio = s->audio_embeds + (size_t) pos * VOX_DEC_DIM;
        float * dst = input_buf + (size_t) pos * VOX_DEC_DIM;
        for (int i = 0; i < VOX_DEC_DIM; i++)
            dst[i] = audio[i] + s->tok_embed_buf[i];
    }

    /* Build decoder graph with n_tokens = prompt_len. */
    const int n_kv = prompt_len;
    const int kv_pos = 0;
    struct ggml_init_params ip = {
        ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false),
        NULL, true,
    };
    struct ggml_context * gctx = ggml_init(ip);

    struct ggml_tensor * ada_arr[26];
    for (int i = 0; i < 26; i++) ada_arr[i] = s->ada.scaled[i];

    vox_decoder_graph_t dg = vox_build_decoder_graph(gctx, s->w, s->dec_kv.layers,
                                                      ada_arr, n_kv, kv_pos, prompt_len);
    ggml_gallocr_t galloc = ggml_gallocr_new(s->buft);
    if (!ggml_gallocr_alloc_graph(galloc, dg.gf)) {
        ggml_gallocr_free(galloc); ggml_free(gctx); free(input_buf);
        fprintf(stderr, "vox_stream: prefill gallocr failed\n");
        return -1;
    }

    ggml_backend_tensor_set(dg.input, input_buf, 0, (size_t) VOX_DEC_DIM * prompt_len * sizeof(float));
    free(input_buf);

    /* Position tensor: [0, 1, 2, ..., prompt_len-1]. */
    int32_t * pos_buf = (int32_t *) malloc(prompt_len * sizeof(int32_t));
    for (int i = 0; i < prompt_len; i++) pos_buf[i] = i;
    ggml_backend_tensor_set(dg.pos, pos_buf, 0, prompt_len * sizeof(int32_t));
    free(pos_buf);

    /* Causal mask: [n_kv, prompt_len]. mask[j, i] = 0 if j <= i, else -INF. */
    float * mask_buf = (float *) malloc((size_t) n_kv * prompt_len * sizeof(float));
    for (int i = 0; i < prompt_len; i++) {
        for (int j = 0; j < n_kv; j++) {
            mask_buf[(size_t) i * n_kv + j] = (j <= i) ? 0.0f : -INFINITY;
        }
    }
    ggml_backend_tensor_set(dg.mask, mask_buf, 0, (size_t) n_kv * prompt_len * sizeof(float));
    free(mask_buf);

    if (ggml_backend_graph_compute(s->backend, dg.gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc); ggml_free(gctx);
        fprintf(stderr, "vox_stream: prefill graph_compute failed\n");
        return -1;
    }

    /* Read logits for the LAST prompt position only (index prompt_len-1).
     * logits tensor shape: [vocab, prompt_len]. We want row prompt_len-1. */
    ggml_backend_tensor_get(dg.logits, s->logits_buf,
        (size_t) (prompt_len - 1) * VOX_VOCAB_SIZE * sizeof(float),
        VOX_VOCAB_SIZE * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(gctx);

    /* Argmax for the first generated token. */
    int argmax = 0;
    float lmax = s->logits_buf[0];
    for (int i = 1; i < VOX_VOCAB_SIZE; i++) {
        if (s->logits_buf[i] > lmax) { lmax = s->logits_buf[i]; argmax = i; }
    }

    if (argmax == TOKEN_EOS) {
        s->eos_seen = 1;
    } else {
        tokens_push(s, argmax);
        s->prev_token = argmax;
    }

    s->decoder_pos = prompt_len;
    VLOG(s, "prefill: %d positions in one shot, first token=%d\n", prompt_len, argmax);
    return 0;
}

/* Advance the streaming pipeline as far as possible:
 *   new mel → conv stem → encoder → adapter → decoder → tokens. */
static int try_advance_pipeline(vox_stream_t * s) {
    if (s->eos_seen) return 0;

    /* 1. Conv stem: mel → conv0 → conv1 → enc_in */
    int n_new_enc = 0;
    float * enc_in = run_conv_stem_stream(s, &n_new_enc);

    /* 2. Encoder step */
    if (n_new_enc > 0 && enc_in) {
        if (run_encoder_step(s, enc_in, n_new_enc) < 0) {
            free(enc_in);
            return -1;
        }
        free(enc_in);
    }

    /* 3. Adapter: consume encoder outputs in groups of 4 */
    while (s->n_enc_total >= (s->n_audio_total + 1) * VOX_DOWNSAMPLE) {
        int enc_start = s->n_audio_total * VOX_DOWNSAMPLE;
        if (run_adapter_step(s, enc_start) < 0) return -1;
    }

    /* 4. Decoder: prefill prompt positions in one shot, then one-at-a-time. */
    int prompt_len = 1 + N_LEFT_PAD_TOKENS + s->opts.delay_tokens;
    if (s->decoder_pos == 0 && s->n_audio_total >= prompt_len && !s->eos_seen) {
        if (run_decoder_prefill(s, prompt_len) < 0) return -1;
    }
    while (s->decoder_pos < s->n_audio_total && !s->eos_seen) {
        if (run_decoder_step_streaming(s) < 0) return -1;
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* Offline pipeline -- Sub-goal A placeholder                         */
/* ------------------------------------------------------------------ */

static int run_offline_pipeline(vox_stream_t * s) {
    if (s->samples_len <= 0) {
        VLOG(s, "vox_stream: no audio samples to process\n");
        return 0;
    }

    /* ---- Pad audio ---- */
    int n_samples_padded = 0;
    float * padded = pad_audio_streaming(s->samples, s->samples_len, &n_samples_padded);
    if (!padded) return -1;
    VLOG(s, "audio padded: %d samples\n", n_samples_padded);

    /* ---- Mel ---- */
    int n_mel_frames = 0;
    float * mel = vox_mel_spectrogram(padded, n_samples_padded, &n_mel_frames);
    free(padded);
    if (!mel) return -1;
    VLOG(s, "mel: %d frames\n", n_mel_frames);

    /* If n_mel_frames is odd, drop the first frame so conv stride=2 works
     * out evenly (matches python_simple_implementation.py). */
    if (n_mel_frames % 2 != 0) {
        memmove(mel, mel + VOX_MEL_BINS, (size_t) (n_mel_frames - 1) * VOX_MEL_BINS * sizeof(float));
        n_mel_frames -= 1;
        VLOG(s, "mel truncated to %d frames\n", n_mel_frames);
    }

    /* ---- Conv stem (CPU) ---- */
    int enc_seq_len = 0;
    float * enc_input = run_conv_stem(s->w, mel, n_mel_frames, &enc_seq_len);
    free(mel);
    if (!enc_input) return -1;
    VLOG(s, "post-conv-stem: enc_seq_len=%d\n", enc_seq_len);

    /* ---- Encoder graph (incremental step builder, called once with
     *      kv_pos=0 and n_new=enc_seq_len -- equivalent to a full forward
     *      pass through an empty cache) ---- */
    if (enc_seq_len > s->max_n_enc) {
        fprintf(stderr, "vox_stream: enc_seq_len %d exceeds enc cache max %d\n",
                enc_seq_len, s->max_n_enc);
        free(enc_input);
        return -1;
    }
    {
        struct ggml_init_params ip = {
            /* .mem_size  = */ ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false),
            /* .mem_buffer = */ NULL,
            /* .no_alloc  = */ true,
        };
        struct ggml_context * gctx = ggml_init(ip);

        vox_encoder_step_graph_t eg = vox_build_encoder_step_graph(
            gctx, s->w, s->enc_kv.layers,
            /* n_new = */ enc_seq_len,
            /* n_total = */ enc_seq_len,
            /* kv_pos = */ 0);
        VLOG(s, "encoder step graph: %d nodes\n", ggml_graph_n_nodes(eg.gf));

        ggml_gallocr_t galloc = ggml_gallocr_new(s->buft);
        if (!ggml_gallocr_alloc_graph(galloc, eg.gf)) {
            ggml_gallocr_free(galloc); ggml_free(gctx); free(enc_input);
            fprintf(stderr, "vox_stream: encoder step gallocr_alloc_graph failed\n");
            return -1;
        }

        ggml_backend_tensor_set(eg.input, enc_input, 0, (size_t) VOX_ENC_DIM * enc_seq_len * sizeof(float));

        int32_t * pos_buf = (int32_t *) malloc(enc_seq_len * sizeof(int32_t));
        for (int i = 0; i < enc_seq_len; i++) pos_buf[i] = i;
        ggml_backend_tensor_set(eg.pos, pos_buf, 0, enc_seq_len * sizeof(int32_t));
        free(pos_buf);

        float * mask_buf = (float *) malloc((size_t) enc_seq_len * enc_seq_len * sizeof(float));
        build_encoder_step_mask(mask_buf, /*n_total*/ enc_seq_len, /*n_new*/ enc_seq_len, /*kv_pos*/ 0);
        ggml_backend_tensor_set(eg.mask, mask_buf, 0, (size_t) enc_seq_len * enc_seq_len * sizeof(float));
        free(mask_buf);

        if (ggml_backend_graph_compute(s->backend, eg.gf) != GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(galloc); ggml_free(gctx); free(enc_input);
            fprintf(stderr, "vox_stream: encoder step graph_compute failed\n");
            return -1;
        }

        float * enc_out = (float *) malloc((size_t) VOX_ENC_DIM * enc_seq_len * sizeof(float));
        ggml_backend_tensor_get(eg.output, enc_out, 0, (size_t) VOX_ENC_DIM * enc_seq_len * sizeof(float));

        ggml_gallocr_free(galloc);
        ggml_free(gctx);
        free(enc_input);
        enc_input = enc_out;
    }

    /* ---- Adapter graph ---- */
    if (enc_seq_len % VOX_DOWNSAMPLE != 0) {
        enc_seq_len -= (enc_seq_len % VOX_DOWNSAMPLE);
        VLOG(s, "encoder output truncated to %d for adapter divisibility\n", enc_seq_len);
    }
    int n_audio = enc_seq_len / VOX_DOWNSAMPLE;
    if (n_audio > s->max_n_audio) {
        fprintf(stderr, "vox_stream: clip exceeds max_audio_seconds (n_audio=%d, max=%d)\n",
                n_audio, s->max_n_audio);
        free(enc_input);
        return -1;
    }

    float * audio_embeds = (float *) malloc((size_t) VOX_DEC_DIM * n_audio * sizeof(float));
    {
        struct ggml_init_params ip = {
            /* .mem_size  = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead(),
            /* .mem_buffer = */ NULL,
            /* .no_alloc  = */ true,
        };
        struct ggml_context * gctx = ggml_init(ip);
        vox_adapter_graph_t ag = vox_build_adapter_graph(gctx, s->w, enc_seq_len);
        ggml_gallocr_t galloc = ggml_gallocr_new(s->buft);
        if (!ggml_gallocr_alloc_graph(galloc, ag.gf)) {
            ggml_gallocr_free(galloc); ggml_free(gctx); free(enc_input); free(audio_embeds);
            fprintf(stderr, "vox_stream: adapter gallocr_alloc_graph failed\n");
            return -1;
        }
        ggml_backend_tensor_set(ag.input, enc_input, 0, (size_t) VOX_ENC_DIM * enc_seq_len * sizeof(float));
        if (ggml_backend_graph_compute(s->backend, ag.gf) != GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(galloc); ggml_free(gctx); free(enc_input); free(audio_embeds);
            fprintf(stderr, "vox_stream: adapter graph_compute failed\n");
            return -1;
        }
        ggml_backend_tensor_get(ag.output, audio_embeds, 0, (size_t) VOX_DEC_DIM * n_audio * sizeof(float));
        ggml_gallocr_free(galloc);
        ggml_free(gctx);
    }
    free(enc_input);
    VLOG(s, "adapter output: n_audio=%d audio embeddings\n", n_audio);

    /* ---- Decoder loop ---- */
    struct ggml_tensor * ada_arr[26];
    for (int i = 0; i < 26; i++) ada_arr[i] = s->ada.scaled[i];

    int prompt_len = 1 + N_LEFT_PAD_TOKENS + s->opts.delay_tokens;
    int prompt[64];
    prompt[0] = TOKEN_BOS;
    for (int i = 1; i < prompt_len; i++) prompt[i] = TOKEN_STREAMING_PAD;

    if (prompt_len > n_audio) {
        fprintf(stderr, "vox_stream: prompt_len (%d) > n_audio (%d), clip too short\n",
                prompt_len, n_audio);
        free(audio_embeds);
        return -1;
    }

    int prev_token = -1;
    float * tok_embed_buf = (float *) malloc(VOX_DEC_DIM * sizeof(float));
    float * input_embed   = (float *) malloc(VOX_DEC_DIM * sizeof(float));
    float * logits_buf    = (float *) malloc(VOX_VOCAB_SIZE * sizeof(float));

    VLOG(s, "decoder loop: %d audio positions, prompt_len=%d\n", n_audio, prompt_len);

    int rc = 0;
    for (int pos = 0; pos < n_audio; pos++) {
        int text_token = (pos < prompt_len) ? prompt[pos] : prev_token;

        read_row_as_f32(s->w->decoder.tok_embeddings, text_token, tok_embed_buf);
        const float * audio = audio_embeds + (size_t) pos * VOX_DEC_DIM;
        for (int i = 0; i < VOX_DEC_DIM; i++) input_embed[i] = audio[i] + tok_embed_buf[i];

        const int n_kv = pos + 1;
        const int kv_pos = pos;
        struct ggml_init_params ip = {
            /* .mem_size  = */ ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false),
            /* .mem_buffer = */ NULL,
            /* .no_alloc  = */ true,
        };
        struct ggml_context * gctx = ggml_init(ip);
        vox_decoder_graph_t dg = vox_build_decoder_graph(gctx, s->w, s->dec_kv.layers, ada_arr, n_kv, kv_pos, 1);
        ggml_gallocr_t galloc = ggml_gallocr_new(s->buft);
        if (!ggml_gallocr_alloc_graph(galloc, dg.gf)) {
            ggml_gallocr_free(galloc); ggml_free(gctx);
            fprintf(stderr, "vox_stream: decoder gallocr_alloc_graph failed (pos=%d)\n", pos);
            rc = -1; break;
        }

        ggml_backend_tensor_set(dg.input, input_embed, 0, VOX_DEC_DIM * sizeof(float));
        int32_t pos_val = pos;
        ggml_backend_tensor_set(dg.pos, &pos_val, 0, sizeof(int32_t));

        float * mask_buf = (float *) calloc(n_kv, sizeof(float));
        ggml_backend_tensor_set(dg.mask, mask_buf, 0, n_kv * sizeof(float));
        free(mask_buf);

        if (ggml_backend_graph_compute(s->backend, dg.gf) != GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(galloc); ggml_free(gctx);
            fprintf(stderr, "vox_stream: decoder graph_compute failed (pos=%d)\n", pos);
            rc = -1; break;
        }

        ggml_backend_tensor_get(dg.logits, logits_buf, 0, VOX_VOCAB_SIZE * sizeof(float));
        ggml_gallocr_free(galloc);
        ggml_free(gctx);

        int argmax = 0;
        float lmax = logits_buf[0];
        for (int i = 1; i < VOX_VOCAB_SIZE; i++) {
            if (logits_buf[i] > lmax) { lmax = logits_buf[i]; argmax = i; }
        }

        /* Generation begins at pos == prompt_len - 1. */
        if (pos >= prompt_len - 1) {
            if (argmax == TOKEN_EOS) {
                VLOG(s, "  pos %d: EOS\n", pos);
                break;
            }
            if (tokens_push(s, argmax) < 0) { rc = -1; break; }
            prev_token = argmax;
        }
    }

    free(tok_embed_buf);
    free(input_embed);
    free(logits_buf);
    free(audio_embeds);
    return rc;
}

/* ------------------------------------------------------------------ */
/* Public API                                                         */
/* ------------------------------------------------------------------ */

vox_stream_opts_t vox_stream_default_opts(void) {
    vox_stream_opts_t o;
    o.backend = 1;             /* CUDA */
    o.delay_tokens = 6;
    o.max_audio_seconds = 30;
    o.verbose = 0;
    o.quant = 0;
    return o;
}

vox_stream_t * vox_stream_init(const char * model_dir, const vox_stream_opts_t * opts_in) {
    if (!model_dir) return NULL;
    vox_stream_opts_t opts = opts_in ? *opts_in : vox_stream_default_opts();

    vox_stream_t * s = (vox_stream_t *) calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->opts = opts;

    ggml_time_init();

    /* Tokenizer */
    char path[1024];
    snprintf(path, sizeof(path), "%s/tekken.json", model_dir);
    if (s->opts.verbose) fprintf(stderr, "loading tokenizer: %s\n", path);
    s->tok = vox_tokenizer_load(path);
    if (!s->tok) { fprintf(stderr, "vox_stream: tokenizer load failed\n"); goto fail; }

    /* Backend */
    if (s->opts.backend == 1) {
        if (ggml_backend_cuda_get_device_count() < 1) {
            fprintf(stderr, "vox_stream: no CUDA device available\n"); goto fail;
        }
        s->backend = ggml_backend_cuda_init(0);
        if (!s->backend) { fprintf(stderr, "vox_stream: ggml_backend_cuda_init failed\n"); goto fail; }
        s->buft = ggml_backend_cuda_buffer_type(0);
        if (s->opts.verbose) {
            char dev_desc[256];
            ggml_backend_cuda_get_device_description(0, dev_desc, sizeof(dev_desc));
            size_t dev_free = 0, dev_total = 0;
            ggml_backend_cuda_get_device_memory(0, &dev_free, &dev_total);
            fprintf(stderr, "CUDA device 0: %s (%.1f / %.1f GB free)\n",
                    dev_desc,
                    dev_free / (1024.0 * 1024.0 * 1024.0),
                    dev_total / (1024.0 * 1024.0 * 1024.0));
        }
    } else {
        s->backend = ggml_backend_cpu_init();
        if (!s->backend) { fprintf(stderr, "vox_stream: ggml_backend_cpu_init failed\n"); goto fail; }
        s->buft = ggml_backend_cpu_buffer_type();
    }

    /* Weights */
    if (s->opts.verbose) fprintf(stderr, "loading weights: %s\n", model_dir);
    enum ggml_type matmul_type = (s->opts.quant == 1) ? GGML_TYPE_Q8_0 : GGML_TYPE_BF16;
    if (s->opts.verbose && matmul_type != GGML_TYPE_BF16)
        fprintf(stderr, "weight quantization: %s\n", ggml_type_name(matmul_type));
    s->w = vox_weights_load(model_dir, s->buft, matmul_type);
    if (!s->w) { fprintf(stderr, "vox_stream: vox_weights_load failed\n"); goto fail; }
    if (s->opts.verbose) {
        fprintf(stderr, "  %d tensors, %.2f GB\n",
                s->w->n_tensors, (double) s->w->n_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    /* ada_scaled */
    if (ada_state_compute(&s->ada, s->w, s->buft, s->opts.delay_tokens) < 0) {
        fprintf(stderr, "vox_stream: ada_state_compute failed\n"); goto fail;
    }
    if (s->opts.verbose) {
        fprintf(stderr, "ada_scaled precomputed for delay_tokens=%d\n", s->opts.delay_tokens);
    }

    /* Decoder KV cache (sized once for max_audio_seconds, reused per clip).
     * Encoder KV cache is sized for max enc positions = 4 * max_n_audio. */
    int max_n_audio = (s->opts.max_audio_seconds * 16000) / RAW_AUDIO_LENGTH_PER_TOK
                      + N_LEFT_PAD_TOKENS + N_RIGHT_PAD_TOKENS + 32;
    s->max_n_audio = max_n_audio;
    s->max_n_enc   = max_n_audio * VOX_DOWNSAMPLE + 16;
    if (kv_cache_alloc(&s->dec_kv, s->buft, max_n_audio) < 0) {
        fprintf(stderr, "vox_stream: kv_cache_alloc failed\n"); goto fail;
    }
    if (enc_kv_cache_alloc(&s->enc_kv, s->buft, s->max_n_enc) < 0) {
        fprintf(stderr, "vox_stream: enc_kv_cache_alloc failed\n"); goto fail;
    }
    if (s->opts.verbose) {
        fprintf(stderr, "dec kv cache: %.2f MB across 26 layers (max_seq=%d)\n",
                (double) ggml_backend_buffer_get_size(s->dec_kv.buf) / (1024.0 * 1024.0),
                s->dec_kv.max_seq);
        fprintf(stderr, "enc kv cache: %.2f MB across 32 layers (max_seq=%d)\n",
                (double) ggml_backend_buffer_get_size(s->enc_kv.buf) / (1024.0 * 1024.0),
                s->enc_kv.max_seq);
    }

    /* Cached F32 conv stem weights (read once from the backend buffer). */
    s->conv0_w = (float *) malloc(ggml_nelements(s->w->encoder.conv0) * sizeof(float));
    s->conv0_b = (float *) malloc(VOX_ENC_DIM * sizeof(float));
    s->conv1_w = (float *) malloc(ggml_nelements(s->w->encoder.conv1) * sizeof(float));
    s->conv1_b = (float *) malloc(VOX_ENC_DIM * sizeof(float));
    if (!s->conv0_w || !s->conv0_b || !s->conv1_w || !s->conv1_b) {
        fprintf(stderr, "vox_stream: conv weight alloc failed\n"); goto fail;
    }
    read_f32_tensor(s->w->encoder.conv0,      s->conv0_w);
    read_f32_tensor(s->w->encoder.conv1,      s->conv1_w);
    read_f32_tensor(s->w->encoder.conv0_bias, s->conv0_b);
    read_f32_tensor(s->w->encoder.conv1_bias, s->conv1_b);

    /* Reusable per-step buffers. */
    s->tok_embed_buf   = (float *) malloc(VOX_DEC_DIM * sizeof(float));
    s->input_embed_buf = (float *) malloc(VOX_DEC_DIM * sizeof(float));
    s->logits_buf      = (float *) malloc(VOX_VOCAB_SIZE * sizeof(float));
    if (!s->tok_embed_buf || !s->input_embed_buf || !s->logits_buf) {
        fprintf(stderr, "vox_stream: per-step buffer alloc failed\n"); goto fail;
    }

    /* Streaming pipeline buffers. */
    s->enc_outputs  = (float *) malloc((size_t) s->max_n_enc   * VOX_ENC_DIM * sizeof(float));
    s->audio_embeds = (float *) malloc((size_t) s->max_n_audio * VOX_DEC_DIM * sizeof(float));
    if (!s->enc_outputs || !s->audio_embeds) {
        fprintf(stderr, "vox_stream: streaming buffer alloc failed\n"); goto fail;
    }

    /* Mel context for incremental mel. */
    s->mel_ctx = vox_mel_ctx_init(N_LEFT_PAD_TOKENS * RAW_AUDIO_LENGTH_PER_TOK);
    if (!s->mel_ctx) { fprintf(stderr, "vox_stream: mel_ctx_init failed\n"); goto fail; }

    /* Legacy sample buffer (for main.c offline path compatibility). */
    s->samples_cap = s->opts.max_audio_seconds * 16000;
    s->samples = (float *) malloc((size_t) s->samples_cap * sizeof(float));
    if (!s->samples) { fprintf(stderr, "vox_stream: samples alloc failed\n"); goto fail; }
    s->samples_len = 0;

    /* Init streaming state counters. */
    s->total_samples_fed = 0;
    s->mel_consumed = 0;
    s->c0_produced  = 0;
    s->c1_produced  = 0;
    s->n_enc_total  = 0;
    s->n_audio_total = 0;
    s->decoder_pos  = 0;
    s->prev_token   = -1;
    s->eos_seen     = 0;
    s->finished     = 0;
    s->mel_tail_count = 0;
    s->c0_tail_count  = 0;

    s->tokens = NULL;
    s->tokens_cap = 0;
    s->tokens_len = 0;
    s->tokens_emitted = 0;

    return s;

fail:
    vox_stream_free(s);
    return NULL;
}

int vox_stream_feed(vox_stream_t * s, const float * samples, int n_samples) {
    if (!s || !samples || n_samples < 0) return -1;
    if (n_samples == 0) return 0;

    /* Feed into incremental mel context. */
    vox_mel_feed(s->mel_ctx, samples, n_samples);
    s->total_samples_fed += n_samples;

    /* Advance the streaming pipeline as far as we can. */
    return try_advance_pipeline(s);
}

int vox_stream_finish(vox_stream_t * s) {
    if (!s || s->finished) return 0;
    s->finished = 1;

    /* Feed the right-pad zeros via mel_feed (NOT mel_finish) so that
     * mel_finish's right-reflect comes from zeros — matching the offline
     * pipeline's pad_audio_streaming + vox_mel_spectrogram reflect pattern. */
    int align_pad = (RAW_AUDIO_LENGTH_PER_TOK - (s->total_samples_fed % RAW_AUDIO_LENGTH_PER_TOK))
                    % RAW_AUDIO_LENGTH_PER_TOK;
    int right_pad = align_pad + N_RIGHT_PAD_TOKENS * RAW_AUDIO_LENGTH_PER_TOK;
    if (right_pad > 0) {
        float * zeros = (float *) calloc(right_pad, sizeof(float));
        if (zeros) {
            vox_mel_feed(s->mel_ctx, zeros, right_pad);
            free(zeros);
        }
    }

    /* Finalize mel context (appends 200-sample right reflect + drops last frame). */
    vox_mel_finish(s->mel_ctx, 0);

    /* Drain all remaining pipeline work. */
    return try_advance_pipeline(s);
}

int vox_stream_get(vox_stream_t * s, vox_token_t * out, int max) {
    if (!s || !out || max <= 0) return 0;
    /* Skip Tekken special-token IDs (< 1000) the same way decode_seq does --
     * the model emits STREAMING_PAD / STREAMING_WORD interleaved with real
     * content, and they should not appear in the streamed text. */
    int n = 0;
    while (n < max && s->tokens_emitted < s->tokens_len) {
        int id = s->tokens[s->tokens_emitted++];
        if (id < 1000) continue;
        out[n].id   = id;
        out[n].text = vox_tokenizer_decode(s->tok, id);
        n++;
    }
    return n;
}

void vox_stream_reset(vox_stream_t * s) {
    if (!s) return;

    /* Reset mel context (free + recreate to clear all internal state). */
    if (s->mel_ctx) vox_mel_free(s->mel_ctx);
    s->mel_ctx = vox_mel_ctx_init(N_LEFT_PAD_TOKENS * RAW_AUDIO_LENGTH_PER_TOK);

    /* Reset streaming pipeline counters. */
    s->total_samples_fed = 0;
    s->mel_consumed   = 0;
    s->c0_produced    = 0;
    s->c1_produced    = 0;
    s->n_enc_total    = 0;
    s->n_audio_total  = 0;
    s->decoder_pos    = 0;
    s->prev_token     = -1;
    s->eos_seen       = 0;
    s->finished       = 0;
    s->mel_tail_count = 0;
    s->c0_tail_count  = 0;

    /* Reset token output. */
    s->tokens_len     = 0;
    s->tokens_emitted = 0;

    /* Legacy. */
    s->samples_len    = 0;
}

void vox_stream_free(vox_stream_t * s) {
    if (!s) return;
    if (s->mel_ctx) vox_mel_free(s->mel_ctx);
    if (s->enc_kv.buf || s->enc_kv.ctx) enc_kv_cache_free(&s->enc_kv);
    if (s->dec_kv.buf || s->dec_kv.ctx) kv_cache_free(&s->dec_kv);
    if (s->ada.buf    || s->ada.ctx)    ada_state_free(&s->ada);
    if (s->w)       vox_weights_free(s->w);
    if (s->backend) ggml_backend_free(s->backend);
    if (s->tok)     vox_tokenizer_free(s->tok);
    free(s->conv0_w);
    free(s->conv0_b);
    free(s->conv1_w);
    free(s->conv1_b);
    free(s->tok_embed_buf);
    free(s->input_embed_buf);
    free(s->logits_buf);
    free(s->enc_outputs);
    free(s->audio_embeds);
    free(s->samples);
    free(s->tokens);
    free(s);
}

void vox_stream_set_delay(vox_stream_t * s, int delay_tokens) {
    if (!s) return;
    s->opts.delay_tokens = delay_tokens;
    ada_state_free(&s->ada);
    ada_state_compute(&s->ada, s->w, s->buft, delay_tokens);
}
