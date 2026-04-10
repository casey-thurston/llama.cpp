/*
 * voxtral-stream main.c - Phase 1F offline transcribe driver.
 *
 * End-to-end pipeline for correctness verification against voxtral.c:
 *   1. Load tokenizer + weights
 *   2. Load WAV, pad it streaming-style (32 left-pad tokens + delay/right pad),
 *      compute mel spectrogram via the vendored voxtral_audio module
 *   3. Conv stem on CPU (vox_causal_conv1d, no ggml)
 *   4. Encoder graph: 32-layer transformer + final RMSNorm
 *   5. Adapter graph: 4x downsample reshape + Linear -> GELU -> Linear
 *   6. Precompute ada_scaled[26] = (1 + ada_up @ gelu(ada_down @ t_cond))
 *      where t_cond = sinusoidal_time_embedding(delay_tokens=6, dim=3072)
 *   7. Decoder loop, one position at a time:
 *        - prefill (positions 0..L-2): input = audio_embed[pos] + tok_embed(prompt[pos])
 *        - first generation (pos=L-1): same form, sample first token
 *        - subsequent (pos=L..n_audio-1): input = audio_embed[pos] + tok_embed(prev)
 *      Stop on EOS.
 *   8. Decode generated tokens to text via the Tekken tokenizer, print.
 *
 * No streaming, no Wyoming, no flags beyond -m / -i. The orchestrator's
 * job here is purely "make voxtral-stream produce a transcript so we can
 * diff it against voxtral.c on the same WAV."
 *
 * Usage: voxtral-stream -m <model_dir> -i <audio.wav>
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

#include "voxtral_tokenizer.h"
#include "voxtral_audio.h"
#include "voxtral_weights.h"
#include "voxtral_model.h"

int vox_verbose = 0;

/* Streaming-style audio padding constants (match python_simple_implementation.py) */
#define RAW_AUDIO_LENGTH_PER_TOK   1280   /* 16000 / 12.5 Hz */
#define N_LEFT_PAD_TOKENS          32     /* * 1280 = 40960 sample left pad */
#define N_DELAY_TOKENS             6      /* 480 ms transcription delay */
#define N_RIGHT_PAD_TOKENS         17     /* (delay+1) + 10 */
#define TOKEN_BOS                  1
#define TOKEN_EOS                  2
#define TOKEN_STREAMING_PAD        32

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

static void die(const char * msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

/* CPU causal conv1d -- offline (run-once-per-clip) port of voxtral.c's
 * vox_causal_conv1d, with the BLAS dependency replaced by manual loops. */
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

/* GELU (matching voxtral.c's vox_gelu and PyTorch's tanh approximation). */
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

/* Read F32 ggml tensor data into a host buffer -- backend-agnostic. */
static void read_f32_tensor(const struct ggml_tensor * t, float * dst) {
    ggml_backend_tensor_get(t, dst, 0, ggml_nbytes(t));
}

/* Read one BF16 row out of a 2D BF16 tensor and convert to F32. The BF16
 * tensor is stored as ne[0]=embed_dim, ne[1]=vocab. row_idx selects which
 * vocab token; output is `embed_dim` floats. Used for tok_embeddings lookup. */
static void read_bf16_row_as_f32(const struct ggml_tensor * t, int row_idx, float * dst) {
    int64_t embed_dim = t->ne[0];
    size_t row_bytes = (size_t) embed_dim * sizeof(uint16_t);
    uint16_t * tmp = (uint16_t *) malloc(row_bytes);
    ggml_backend_tensor_get(t, tmp, (size_t) row_idx * row_bytes, row_bytes);
    for (int64_t i = 0; i < embed_dim; i++) {
        /* BF16 -> F32: prepend low 16 bits with zero. */
        uint32_t bits = ((uint32_t) tmp[i]) << 16;
        float f;
        memcpy(&f, &bits, sizeof(f));
        dst[i] = f;
    }
    free(tmp);
}

/* ------------------------------------------------------------------ */
/* ada_scaled precompute                                              */
/* ------------------------------------------------------------------ */

/* Compute (1 + ada_up @ gelu(ada_down @ t_cond)) for each of the 26 decoder
 * layers, store as F32 ggml_tensors in their own backend buffer. */
typedef struct {
    struct ggml_context  * ctx;
    ggml_backend_buffer_t  buf;
    struct ggml_tensor   * scaled[26];
} ada_state_t;

static void ada_state_compute(ada_state_t * a, const vox_weights_t * w,
                              ggml_backend_buffer_type_t buft, int delay_tokens) {
    /* Allocate context + tensors */
    struct ggml_init_params ip = {
        /* .mem_size  = */ ggml_tensor_overhead() * 64,
        /* .mem_buffer = */ NULL,
        /* .no_alloc  = */ true,
    };
    a->ctx = ggml_init(ip);
    char name[32];
    for (int i = 0; i < 26; i++) {
        a->scaled[i] = ggml_new_tensor_1d(a->ctx, GGML_TYPE_F32, VOX_DEC_DIM);
        snprintf(name, sizeof(name), "ada_scaled.%d", i);
        ggml_set_name(a->scaled[i], name);
    }
    a->buf = ggml_backend_alloc_ctx_tensors_from_buft(a->ctx, buft);
    if (!a->buf) die("ada_state: backend buffer alloc failed");

    /* Compute t_cond on CPU */
    float t_cond[VOX_DEC_DIM];
    cpu_compute_time_embedding(t_cond, (float) delay_tokens, VOX_DEC_DIM);

    /* Per-layer F32 work buffers for ada_norm_down (3072->32) and ada_norm_up (32->3072). */
    float * down_w = (float *) malloc(VOX_DEC_DIM * 32 * sizeof(float));
    float * up_w   = (float *) malloc(32 * VOX_DEC_DIM * sizeof(float));
    float hidden[32];
    float scaled[VOX_DEC_DIM];

    for (int i = 0; i < 26; i++) {
        const vox_dec_layer_w_t * lw = &w->decoder.layers[i];

        /* ada_norm_down ggml shape: ne[0]=3072, ne[1]=32 (in,out). Stored
         * row-major: down_w[i*3072 + j] = down(out=i, in=j). */
        read_f32_tensor(lw->ada_norm_down, down_w);

        /* hidden = ada_norm_down @ t_cond  -> [32] */
        for (int oi = 0; oi < 32; oi++) {
            float acc = 0.0f;
            const float * row = down_w + (size_t) oi * VOX_DEC_DIM;
            for (int j = 0; j < VOX_DEC_DIM; j++) acc += row[j] * t_cond[j];
            hidden[oi] = acc;
        }
        cpu_gelu_inplace(hidden, 32);

        /* ada_norm_up ggml shape: ne[0]=32, ne[1]=3072. up_w[oi*32 + j] = up(out=oi, in=j). */
        read_f32_tensor(lw->ada_norm_up, up_w);

        /* scaled = 1 + ada_norm_up @ hidden -> [3072] */
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
}

static void ada_state_free(ada_state_t * a) {
    if (a->buf) ggml_backend_buffer_free(a->buf);
    if (a->ctx) ggml_free(a->ctx);
    memset(a, 0, sizeof(*a));
}

/* ------------------------------------------------------------------ */
/* KV cache                                                           */
/* ------------------------------------------------------------------ */

typedef struct {
    struct ggml_context  * ctx;
    ggml_backend_buffer_t  buf;
    int                    max_seq;
    vox_kv_cache_layer_t   layers[26];
} kv_cache_t;

static void kv_cache_alloc(kv_cache_t * c, ggml_backend_buffer_type_t buft, int max_seq) {
    struct ggml_init_params ip = {
        /* .mem_size  = */ ggml_tensor_overhead() * 64,
        /* .mem_buffer = */ NULL,
        /* .no_alloc  = */ true,
    };
    c->ctx = ggml_init(ip);
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
    if (!c->buf) die("kv_cache: backend buffer alloc failed");
}

static void kv_cache_free(kv_cache_t * c) {
    if (c->buf) ggml_backend_buffer_free(c->buf);
    if (c->ctx) ggml_free(c->ctx);
    memset(c, 0, sizeof(*c));
}

/* ------------------------------------------------------------------ */
/* Pipeline stages                                                    */
/* ------------------------------------------------------------------ */

/* Pad audio streaming-style: prepend N_LEFT_PAD_TOKENS*1280 zeros, append
 * align-to-1280 zeros plus N_RIGHT_PAD_TOKENS*1280 zeros. Returns new
 * allocated buffer; caller frees. */
static float * pad_audio_streaming(const float * in, int n_in, int * n_out_ptr) {
    int align_pad = (RAW_AUDIO_LENGTH_PER_TOK - (n_in % RAW_AUDIO_LENGTH_PER_TOK)) % RAW_AUDIO_LENGTH_PER_TOK;
    int right_pad = align_pad + N_RIGHT_PAD_TOKENS * RAW_AUDIO_LENGTH_PER_TOK;
    int left_pad  = N_LEFT_PAD_TOKENS * RAW_AUDIO_LENGTH_PER_TOK;
    int n_out = left_pad + n_in + right_pad;
    float * out = (float *) calloc(n_out, sizeof(float));
    memcpy(out + left_pad, in, n_in * sizeof(float));
    *n_out_ptr = n_out;
    return out;
}

/* Run the conv stem in plain C and produce the [VOX_ENC_DIM, conv_seq_len]
 * F32 input that the encoder graph expects.
 *
 * Mel comes from voxtral_audio as [n_mel_frames, 128] row-major. We need
 * [128, n_mel_frames] for the conv1d input.
 *
 * conv0: [128, n_mel] -> [1280, n_mel]   (kernel=3, stride=1)
 * GELU
 * conv1: [1280, n_mel] -> [1280, n_mel/2] (kernel=3, stride=2)
 * GELU
 *
 * Returns [1280, conv_seq_len] with conv_seq_len set in *out_seq_len_ptr.
 * Caller frees the returned buffer. */
static float * run_conv_stem(
    const vox_weights_t * w,
    const float * mel_t_first,   /* [n_mel_frames, 128] row-major (out of vox_mel_spectrogram) */
    int n_mel_frames,
    int * out_seq_len_ptr)
{
    /* Read conv0 / conv1 weights + biases out of the backend buffer into
     * plain F32 arrays we can iterate. (They were loaded as F32 in 1C.) */
    int conv0_w_n = (int) ggml_nelements(w->encoder.conv0);   /* 1280*128*3 */
    int conv1_w_n = (int) ggml_nelements(w->encoder.conv1);   /* 1280*1280*3 */
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

    /* Conv0: [128, n_mel] -> [1280, conv0_out] */
    int conv0_out = 0;
    float * after_conv0 = cpu_causal_conv1d(
        conv_in, conv0_w, conv0_b,
        VOX_MEL_BINS, VOX_ENC_DIM, n_mel_frames,
        /*kernel*/ 3, /*stride*/ 1, &conv0_out);
    cpu_gelu_inplace(after_conv0, VOX_ENC_DIM * conv0_out);
    free(conv_in);
    free(conv0_w);
    free(conv0_b);

    /* Conv1: [1280, conv0_out] -> [1280, conv1_out] */
    int conv1_out = 0;
    float * after_conv1 = cpu_causal_conv1d(
        after_conv0, conv1_w, conv1_b,
        VOX_ENC_DIM, VOX_ENC_DIM, conv0_out,
        /*kernel*/ 3, /*stride*/ 2, &conv1_out);
    cpu_gelu_inplace(after_conv1, VOX_ENC_DIM * conv1_out);
    free(after_conv0);
    free(conv1_w);
    free(conv1_b);

    /* The encoder graph expects [enc_dim, n_pos] in ggml innermost-first
     * order. cpu_causal_conv1d output is [channels_out, out_length] row-major,
     * which is exactly that layout (ne[0]=out_length=conv1_out is innermost).
     *
     * Wait -- ggml innermost-first means ne[0] is the contiguous (fastest)
     * dim. For ggml [enc_dim, n_pos] = (ne[0]=enc_dim, ne[1]=n_pos), the
     * memory layout is "all enc_dim values for pos 0, then all enc_dim values
     * for pos 1, ...". Our after_conv1 layout is "all out_length values for
     * channel 0, then all for channel 1, ..." -- which means CHANNEL is the
     * outer dim and POSITION is the inner dim. That's the OPPOSITE of what
     * ggml wants. We need to transpose. */
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

/* Build the causal mask for the encoder. Output buffer is [n_pos, n_pos]
 * F32 row-major; element (i, j) is 0 if j is reachable from i, else -INF.
 * (Sliding window of VOX_ENC_WINDOW=750.) */
static void build_encoder_mask(float * mask, int n_pos) {
    const int window = VOX_ENC_WINDOW;
    for (int i = 0; i < n_pos; i++) {
        for (int j = 0; j < n_pos; j++) {
            /* Causal: j > i is masked. Sliding window: j < i - window + 1 is masked. */
            int allowed = (j <= i) && (j >= i - window + 1);
            mask[(size_t) i * n_pos + j] = allowed ? 0.0f : -INFINITY;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */

static void usage(const char * argv0) {
    fprintf(stderr, "usage: %s -m <model_dir> -i <audio.wav>\n", argv0);
}

int main(int argc, char ** argv) {
    const char * model_dir = NULL;
    const char * wav_path  = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)      model_dir = argv[++i];
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) wav_path  = argv[++i];
        else if (strcmp(argv[i], "-v") == 0)                  vox_verbose = 1;
        else { usage(argv[0]); return 2; }
    }
    if (!model_dir || !wav_path) { usage(argv[0]); return 2; }

    ggml_time_init();
    fprintf(stderr, "voxtral-stream: ggml link OK (t=%lld us)\n", (long long) ggml_time_us());

    /* ---- Tokenizer ---- */
    char path[1024];
    snprintf(path, sizeof(path), "%s/tekken.json", model_dir);
    fprintf(stderr, "loading tokenizer: %s\n", path);
    vox_tokenizer_t * tok = vox_tokenizer_load(path);
    if (!tok) die("tokenizer load failed");

    /* ---- CUDA backend ---- */
    if (ggml_backend_cuda_get_device_count() < 1) die("no CUDA device available");
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) die("ggml_backend_cuda_init failed");
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(0);
    {
        char dev_desc[256];
        ggml_backend_cuda_get_device_description(0, dev_desc, sizeof(dev_desc));
        size_t dev_free = 0, dev_total = 0;
        ggml_backend_cuda_get_device_memory(0, &dev_free, &dev_total);
        fprintf(stderr, "CUDA device 0: %s (%.1f / %.1f GB free)\n",
                dev_desc, dev_free / (1024.0 * 1024.0 * 1024.0),
                dev_total / (1024.0 * 1024.0 * 1024.0));
    }

    /* ---- Weights (CUDA backend buffer) ---- */
    fprintf(stderr, "loading weights: %s\n", model_dir);
    vox_weights_t * w = vox_weights_load(model_dir, buft);
    if (!w) die("vox_weights_load failed");
    fprintf(stderr, "  %d tensors, %.2f GB on CUDA\n",
            w->n_tensors, (double) w->n_bytes / (1024.0 * 1024.0 * 1024.0));

    /* ---- WAV + mel ---- */
    int n_samples_raw = 0;
    float * raw = vox_load_wav(wav_path, &n_samples_raw);
    if (!raw) die("vox_load_wav failed");
    fprintf(stderr, "audio: %d samples (%.2f s)\n", n_samples_raw, n_samples_raw / 16000.0);

    int n_samples_padded = 0;
    float * padded = pad_audio_streaming(raw, n_samples_raw, &n_samples_padded);
    free(raw);
    fprintf(stderr, "audio padded: %d samples\n", n_samples_padded);

    int n_mel_frames = 0;
    float * mel = vox_mel_spectrogram(padded, n_samples_padded, &n_mel_frames);
    free(padded);
    if (!mel) die("vox_mel_spectrogram failed");
    fprintf(stderr, "mel: %d frames\n", n_mel_frames);

    /* If n_mel_frames is odd, drop the first frame so conv stride=2 works
     * out evenly (matches python_simple_implementation.py). */
    if (n_mel_frames % 2 != 0) {
        memmove(mel, mel + VOX_MEL_BINS, (size_t) (n_mel_frames - 1) * VOX_MEL_BINS * sizeof(float));
        n_mel_frames -= 1;
        fprintf(stderr, "mel truncated to %d frames\n", n_mel_frames);
    }

    /* ---- Conv stem (CPU) ---- */
    int enc_seq_len = 0;
    float * enc_input = run_conv_stem(w, mel, n_mel_frames, &enc_seq_len);
    free(mel);
    fprintf(stderr, "post-conv-stem: enc_seq_len=%d\n", enc_seq_len);

    /* ---- Encoder graph ---- */
    {
        struct ggml_init_params ip = {
            /* .mem_size  = */ ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false),
            /* .mem_buffer = */ NULL,
            /* .no_alloc  = */ true,
        };
        struct ggml_context * gctx = ggml_init(ip);

        vox_encoder_graph_t eg = vox_build_encoder_graph(gctx, w, enc_seq_len);
        fprintf(stderr, "encoder graph: %d nodes\n", ggml_graph_n_nodes(eg.gf));

        ggml_gallocr_t galloc = ggml_gallocr_new(buft);
        if (!ggml_gallocr_alloc_graph(galloc, eg.gf)) die("encoder gallocr_alloc_graph failed");
        fprintf(stderr, "  compute buffer: %.2f MB\n",
                (double) ggml_gallocr_get_buffer_size(galloc, 0) / (1024.0 * 1024.0));

        /* Fill input + position + mask */
        ggml_backend_tensor_set(eg.input, enc_input, 0, (size_t) VOX_ENC_DIM * enc_seq_len * sizeof(float));

        int32_t * pos_buf = (int32_t *) malloc(enc_seq_len * sizeof(int32_t));
        for (int i = 0; i < enc_seq_len; i++) pos_buf[i] = i;
        ggml_backend_tensor_set(eg.pos, pos_buf, 0, enc_seq_len * sizeof(int32_t));
        free(pos_buf);

        float * mask_buf = (float *) malloc((size_t) enc_seq_len * enc_seq_len * sizeof(float));
        build_encoder_mask(mask_buf, enc_seq_len);
        ggml_backend_tensor_set(eg.mask, mask_buf, 0, (size_t) enc_seq_len * enc_seq_len * sizeof(float));
        free(mask_buf);

        if (ggml_backend_graph_compute(backend, eg.gf) != GGML_STATUS_SUCCESS)
            die("encoder graph_compute failed");

        /* Read encoder output, save to a heap buffer for the adapter graph. */
        float * enc_out = (float *) malloc((size_t) VOX_ENC_DIM * enc_seq_len * sizeof(float));
        ggml_backend_tensor_get(eg.output, enc_out, 0, (size_t) VOX_ENC_DIM * enc_seq_len * sizeof(float));

        ggml_gallocr_free(galloc);
        ggml_free(gctx);
        free(enc_input);
        enc_input = enc_out; /* hand off to adapter stage */
    }

    /* ---- Adapter graph ---- */
    /* enc_input is [enc_dim=1280, enc_seq_len], adapter wants enc_seq_len % 4 == 0. */
    if (enc_seq_len % VOX_DOWNSAMPLE != 0) {
        /* Truncate from the right to fit. */
        enc_seq_len -= (enc_seq_len % VOX_DOWNSAMPLE);
        fprintf(stderr, "encoder output truncated to %d for adapter divisibility\n", enc_seq_len);
    }
    int n_audio = enc_seq_len / VOX_DOWNSAMPLE;
    float * audio_embeds = (float *) malloc((size_t) VOX_DEC_DIM * n_audio * sizeof(float));
    {
        struct ggml_init_params ip = {
            /* .mem_size  = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead(),
            /* .mem_buffer = */ NULL,
            /* .no_alloc  = */ true,
        };
        struct ggml_context * gctx = ggml_init(ip);
        vox_adapter_graph_t ag = vox_build_adapter_graph(gctx, w, enc_seq_len);
        ggml_gallocr_t galloc = ggml_gallocr_new(buft);
        if (!ggml_gallocr_alloc_graph(galloc, ag.gf)) die("adapter gallocr_alloc_graph failed");
        ggml_backend_tensor_set(ag.input, enc_input, 0, (size_t) VOX_ENC_DIM * enc_seq_len * sizeof(float));
        if (ggml_backend_graph_compute(backend, ag.gf) != GGML_STATUS_SUCCESS)
            die("adapter graph_compute failed");
        ggml_backend_tensor_get(ag.output, audio_embeds, 0, (size_t) VOX_DEC_DIM * n_audio * sizeof(float));
        ggml_gallocr_free(galloc);
        ggml_free(gctx);
    }
    free(enc_input);
    fprintf(stderr, "adapter output: n_audio=%d audio embeddings\n", n_audio);

    /* ---- ada_scaled precompute ---- */
    ada_state_t ada = {0};
    ada_state_compute(&ada, w, buft, N_DELAY_TOKENS);
    struct ggml_tensor * ada_arr[26];
    for (int i = 0; i < 26; i++) ada_arr[i] = ada.scaled[i];
    fprintf(stderr, "ada_scaled precomputed for delay_tokens=%d\n", N_DELAY_TOKENS);

    /* ---- KV cache (offline path needs n_audio positions) ---- */
    kv_cache_t kv = {0};
    kv_cache_alloc(&kv, buft, n_audio + 32);
    fprintf(stderr, "kv cache: %.2f MB across 26 layers (max_seq=%d)\n",
            (double) ggml_backend_buffer_get_size(kv.buf) / (1024.0 * 1024.0), kv.max_seq);

    /* ---- Decoder loop ---- */
    /* prompt: [BOS] + STREAMING_PAD * (32 + 6) = 39 tokens */
    int prompt_len = 1 + N_LEFT_PAD_TOKENS + N_DELAY_TOKENS;
    int prompt[64];
    prompt[0] = TOKEN_BOS;
    for (int i = 1; i < prompt_len; i++) prompt[i] = TOKEN_STREAMING_PAD;

    if (prompt_len > n_audio) {
        fprintf(stderr, "ERROR: prompt_len (%d) > n_audio (%d) -- clip too short\n",
                prompt_len, n_audio);
        return 1;
    }

    int generated[2048];
    int n_generated = 0;
    int prev_token = -1;

    fprintf(stderr, "decoder loop: %d audio positions, prompt_len=%d\n", n_audio, prompt_len);

    float * tok_embed_buf = (float *) malloc(VOX_DEC_DIM * sizeof(float));
    float * input_embed   = (float *) malloc(VOX_DEC_DIM * sizeof(float));
    float * logits_buf    = (float *) malloc(VOX_VOCAB_SIZE * sizeof(float));

    for (int pos = 0; pos < n_audio; pos++) {
        /* Determine which text token's embedding gets summed with audio_embed[pos]. */
        int text_token;
        if (pos < prompt_len) {
            text_token = prompt[pos];
        } else {
            text_token = prev_token; /* previously generated token */
        }

        /* Look up tok_embed(text_token) and add to audio_embed[pos]. */
        read_bf16_row_as_f32(w->decoder.tok_embeddings, text_token, tok_embed_buf);
        const float * audio = audio_embeds + (size_t) pos * VOX_DEC_DIM;
        for (int i = 0; i < VOX_DEC_DIM; i++) input_embed[i] = audio[i] + tok_embed_buf[i];

        /* Build the decoder graph for this step. */
        const int n_kv = pos + 1;
        const int kv_pos = pos;
        struct ggml_init_params ip = {
            /* .mem_size  = */ ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false),
            /* .mem_buffer = */ NULL,
            /* .no_alloc  = */ true,
        };
        struct ggml_context * gctx = ggml_init(ip);
        vox_decoder_graph_t dg = vox_build_decoder_graph(gctx, w, kv.layers, ada_arr, n_kv, kv_pos);
        ggml_gallocr_t galloc = ggml_gallocr_new(buft);
        if (!ggml_gallocr_alloc_graph(galloc, dg.gf)) die("decoder gallocr_alloc_graph failed");

        ggml_backend_tensor_set(dg.input, input_embed, 0, VOX_DEC_DIM * sizeof(float));
        int32_t pos_val = pos;
        ggml_backend_tensor_set(dg.pos, &pos_val, 0, sizeof(int32_t));

        /* Mask: all-zero for n_kv positions (decoder is causal but we only have
         * one new token per step, and it can attend to everything in the cache). */
        float * mask_buf = (float *) calloc(n_kv, sizeof(float));
        ggml_backend_tensor_set(dg.mask, mask_buf, 0, n_kv * sizeof(float));
        free(mask_buf);

        if (ggml_backend_graph_compute(backend, dg.gf) != GGML_STATUS_SUCCESS)
            die("decoder graph_compute failed");

        ggml_backend_tensor_get(dg.logits, logits_buf, 0, VOX_VOCAB_SIZE * sizeof(float));

        ggml_gallocr_free(galloc);
        ggml_free(gctx);

        /* Argmax */
        int argmax = 0;
        float lmax = logits_buf[0];
        for (int i = 1; i < VOX_VOCAB_SIZE; i++) {
            if (logits_buf[i] > lmax) { lmax = logits_buf[i]; argmax = i; }
        }

        /* Generation begins at pos == prompt_len - 1 (the last prompt position
         * gives us our first generated token), then continues for pos >= prompt_len. */
        if (pos >= prompt_len - 1) {
            if (argmax == TOKEN_EOS) {
                fprintf(stderr, "  pos %d: EOS\n", pos);
                break;
            }
            generated[n_generated++] = argmax;
            prev_token = argmax;
            if (n_generated <= 5 || n_generated % 10 == 0) {
                const char * s = vox_tokenizer_decode(tok, argmax);
                fprintf(stderr, "  pos %d: token %d (%s) logit %.3f\n",
                        pos, argmax, s ? s : "<null>", lmax);
            }
        }
    }

    free(tok_embed_buf);
    free(input_embed);
    free(logits_buf);
    free(audio_embeds);

    /* ---- Decode tokens to text and print ---- */
    char * text = vox_tokenizer_decode_seq(tok, generated, n_generated);
    if (text) {
        printf("%s\n", text);
        free(text);
    }

    fprintf(stderr, "\ngenerated %d tokens\n", n_generated);

    kv_cache_free(&kv);
    ada_state_free(&ada);
    ggml_backend_free(backend);
    vox_weights_free(w);
    vox_tokenizer_free(tok);

    return 0;
}
