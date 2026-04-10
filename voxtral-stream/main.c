/*
 * voxtral-stream main.c - Phase 1D smoke test.
 *
 * Builds on the 1C smoke test (vox_weights_load) by:
 *   - Building the adapter graph against synthetic encoder hidden states,
 *     executing it on a CPU backend, and printing the output shape and a
 *     few values.
 *   - Allocating a small per-layer KV cache, building the single-token
 *     decoder graph for pos=0, executing it, and printing the logits
 *     argmax + a few sample values.
 *
 * The 1D smoke test does NOT verify numerical correctness against
 * voxtral.c -- the input is synthetic and ada_scale is omitted (equivalent
 * to delay_tokens=0). It verifies that the graph builders produce ggml
 * topologies that allocate, schedule, and run to completion without
 * shape/dtype errors. End-to-end correctness against voxtral.c is 1F.
 *
 * Usage:
 *   voxtral-stream <model_dir>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "voxtral_tokenizer.h"
#include "voxtral_audio.h"
#include "voxtral_weights.h"
#include "voxtral_model.h"

int vox_verbose = 0;

static void print_usage(const char * argv0) {
    fprintf(stderr, "usage: %s <model_dir>\n", argv0);
    fprintf(stderr, "  <model_dir> must contain consolidated.safetensors and tekken.json\n");
}

static void print_shape(const char * label, const struct ggml_tensor * t) {
    if (!t) {
        fprintf(stderr, "  [MISSING] %s\n", label);
        return;
    }
    const char * dtype = (t->type == GGML_TYPE_BF16) ? "bf16" :
                         (t->type == GGML_TYPE_F32)  ? "f32 " :
                         (t->type == GGML_TYPE_F16)  ? "f16 " : "????";
    fprintf(stderr, "  [%s] %s shape=[", dtype, label);
    for (int d = 0; d < ggml_n_dims(t); d++) {
        fprintf(stderr, "%s%lld", d ? "," : "", (long long) t->ne[d]);
    }
    fprintf(stderr, "]\n");
}

/* ------------------------------------------------------------------ */
/* Adapter smoke test                                                 */
/* ------------------------------------------------------------------ */

static int run_adapter_smoke(const vox_weights_t * w, ggml_backend_t backend) {
    fprintf(stderr, "\n--- adapter smoke test ---\n");

    /* Build the adapter graph: 8 encoder positions -> 8/4 = 2 decoder positions */
    const int n_pos = 8;

    struct ggml_init_params ip = {
        /* .mem_size  = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc  = */ true,
    };
    struct ggml_context * gctx = ggml_init(ip);

    vox_adapter_graph_t g = vox_build_adapter_graph(gctx, w, n_pos);
    print_shape("adapter.input ", g.input);
    print_shape("adapter.output", g.output);
    fprintf(stderr, "  graph nodes = %d\n", ggml_graph_n_nodes(g.gf));

    /* Allocate intermediate tensors for the graph on the CPU backend buffer. */
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!ggml_gallocr_alloc_graph(galloc, g.gf)) {
        fprintf(stderr, "ERROR: ggml_gallocr_alloc_graph failed (adapter)\n");
        ggml_gallocr_free(galloc);
        ggml_free(gctx);
        return -1;
    }

    /* Fill input with a small deterministic pattern. */
    const int input_n = (int) ggml_nelements(g.input);
    float * in_data = (float *) malloc(input_n * sizeof(float));
    for (int i = 0; i < input_n; i++) in_data[i] = 0.001f * (float) (i % 100);
    ggml_backend_tensor_set(g.input, in_data, 0, input_n * sizeof(float));
    free(in_data);

    /* Compute */
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: ggml_backend_graph_compute failed (adapter)\n");
        ggml_gallocr_free(galloc);
        ggml_free(gctx);
        return -1;
    }

    /* Read output */
    const int out_n = (int) ggml_nelements(g.output);
    float * out_data = (float *) malloc(out_n * sizeof(float));
    ggml_backend_tensor_get(g.output, out_data, 0, out_n * sizeof(float));
    fprintf(stderr, "  adapter output[0..7] =");
    for (int i = 0; i < 8 && i < out_n; i++) fprintf(stderr, " %.4f", out_data[i]);
    fprintf(stderr, "\n  adapter output finite-check: ");
    int n_nan = 0, n_inf = 0;
    for (int i = 0; i < out_n; i++) {
        if (out_data[i] != out_data[i]) n_nan++;
        if (out_data[i] >  1e30f || out_data[i] < -1e30f) n_inf++;
    }
    fprintf(stderr, "%d nan, %d inf, %d total\n", n_nan, n_inf, out_n);
    free(out_data);

    ggml_gallocr_free(galloc);
    ggml_free(gctx);

    return (n_nan == 0 && n_inf == 0) ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* Decoder smoke test                                                 */
/* ------------------------------------------------------------------ */

#define KV_MAX_SEQ 16  /* small cache for the smoke test */

typedef struct {
    struct ggml_context  * ctx;
    ggml_backend_buffer_t  buf;
    vox_kv_cache_layer_t   layers[26]; /* must match VOX_DEC_LAYERS */
} dec_kv_cache_t;

static int dec_kv_cache_alloc(dec_kv_cache_t * c) {
    struct ggml_init_params ip = {
        /* .mem_size  = */ ggml_tensor_overhead() * 64,
        /* .mem_buffer = */ NULL,
        /* .no_alloc  = */ true,
    };
    c->ctx = ggml_init(ip);
    if (!c->ctx) return -1;

    /* 2D layout per layer: [head_dim*n_kv_heads, max_seq] -- matches llama.cpp's
     * llama_kv_cache. The decoder builder reshapes+permutes this for attention. */
    const int kv_dim = VOX_DEC_HEAD_DIM * VOX_DEC_KV_HEADS;
    char name[64];
    for (int i = 0; i < 26; i++) {
        c->layers[i].k = ggml_new_tensor_2d(c->ctx, GGML_TYPE_F32, kv_dim, KV_MAX_SEQ);
        snprintf(name, sizeof(name), "kv.k.%d", i);
        ggml_set_name(c->layers[i].k, name);

        c->layers[i].v = ggml_new_tensor_2d(c->ctx, GGML_TYPE_F32, kv_dim, KV_MAX_SEQ);
        snprintf(name, sizeof(name), "kv.v.%d", i);
        ggml_set_name(c->layers[i].v, name);
    }

    c->buf = ggml_backend_alloc_ctx_tensors_from_buft(c->ctx, ggml_backend_cpu_buffer_type());
    if (!c->buf) {
        ggml_free(c->ctx);
        c->ctx = NULL;
        return -1;
    }
    return 0;
}

static void dec_kv_cache_free(dec_kv_cache_t * c) {
    if (c->buf) ggml_backend_buffer_free(c->buf);
    if (c->ctx) ggml_free(c->ctx);
    c->buf = NULL;
    c->ctx = NULL;
}

static int run_decoder_smoke(const vox_weights_t * w, ggml_backend_t backend) {
    fprintf(stderr, "\n--- decoder smoke test ---\n");

    /* Allocate small KV cache (single position is enough for the first step). */
    dec_kv_cache_t cache = {0};
    if (dec_kv_cache_alloc(&cache) != 0) {
        fprintf(stderr, "ERROR: dec_kv_cache_alloc failed\n");
        return -1;
    }
    fprintf(stderr, "  KV cache: %.2f MB across 26 layers\n",
            (double) ggml_backend_buffer_get_size(cache.buf) / (1024.0 * 1024.0));

    /* Build the decoder graph for pos=0, n_kv=1, kv_pos=0. */
    const int n_kv   = 1;
    const int kv_pos = 0;

    struct ggml_init_params ip = {
        /* .mem_size  = */ ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc  = */ true,
    };
    struct ggml_context * gctx = ggml_init(ip);

    vox_decoder_graph_t g = vox_build_decoder_graph(gctx, w, cache.layers, n_kv, kv_pos);
    print_shape("decoder.input ", g.input);
    print_shape("decoder.pos   ", g.pos);
    print_shape("decoder.mask  ", g.mask);
    print_shape("decoder.logits", g.logits);
    fprintf(stderr, "  graph nodes = %d\n", ggml_graph_n_nodes(g.gf));

    /* Allocate intermediate tensors */
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!ggml_gallocr_alloc_graph(galloc, g.gf)) {
        fprintf(stderr, "ERROR: ggml_gallocr_alloc_graph failed (decoder)\n");
        ggml_gallocr_free(galloc);
        ggml_free(gctx);
        dec_kv_cache_free(&cache);
        return -1;
    }
    fprintf(stderr, "  compute buffer = %.2f MB\n",
            (double) ggml_gallocr_get_buffer_size(galloc, 0) / (1024.0 * 1024.0));

    /* Fill inputs */
    float * input_data = (float *) calloc(VOX_DEC_DIM, sizeof(float));
    for (int i = 0; i < VOX_DEC_DIM; i++) input_data[i] = 0.01f * sinf(0.001f * (float) i);
    ggml_backend_tensor_set(g.input, input_data, 0, VOX_DEC_DIM * sizeof(float));
    free(input_data);

    int32_t pos_val = 0;
    ggml_backend_tensor_set(g.pos, &pos_val, 0, sizeof(int32_t));

    float mask_val = 0.0f; /* the new token can attend to itself, no masking */
    ggml_backend_tensor_set(g.mask, &mask_val, 0, sizeof(float));

    /* Compute */
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: ggml_backend_graph_compute failed (decoder)\n");
        ggml_gallocr_free(galloc);
        ggml_free(gctx);
        dec_kv_cache_free(&cache);
        return -1;
    }

    /* Read logits */
    float * logits = (float *) malloc(VOX_VOCAB_SIZE * sizeof(float));
    ggml_backend_tensor_get(g.logits, logits, 0, VOX_VOCAB_SIZE * sizeof(float));

    /* Argmax + finite check */
    int argmax = 0;
    float lmax = logits[0];
    int n_nan = 0, n_inf = 0;
    for (int i = 0; i < VOX_VOCAB_SIZE; i++) {
        if (logits[i] != logits[i]) { n_nan++; continue; }
        if (logits[i] > 1e30f || logits[i] < -1e30f) { n_inf++; continue; }
        if (logits[i] > lmax) { lmax = logits[i]; argmax = i; }
    }
    fprintf(stderr, "  decoder logits[0..7] =");
    for (int i = 0; i < 8; i++) fprintf(stderr, " %.4f", logits[i]);
    fprintf(stderr, "\n  decoder argmax token = %d (logit %.4f)\n", argmax, lmax);
    fprintf(stderr, "  decoder logits finite-check: %d nan, %d inf, %d total\n",
            n_nan, n_inf, VOX_VOCAB_SIZE);
    free(logits);

    ggml_gallocr_free(galloc);
    ggml_free(gctx);
    dec_kv_cache_free(&cache);

    return (n_nan == 0 && n_inf == 0) ? 0 : -1;
}

/* ------------------------------------------------------------------ */

int main(int argc, char ** argv) {
    ggml_time_init();
    fprintf(stderr, "voxtral-stream: ggml link OK (t=%lld us)\n",
            (long long) ggml_time_us());

    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    const char * model_dir = argv[1];
    char path[1024];

    /* Tokenizer */
    snprintf(path, sizeof(path), "%s/tekken.json", model_dir);
    fprintf(stderr, "loading tokenizer: %s\n", path);
    vox_tokenizer_t * tok = vox_tokenizer_load(path);
    if (!tok) { fprintf(stderr, "ERROR: failed to load tokenizer\n"); return 1; }
    fprintf(stderr, "tokenizer OK: vocab=%d bos=%d eos=%d\n",
            vox_tokenizer_vocab_size(tok),
            vox_tokenizer_bos(tok),
            vox_tokenizer_eos(tok));

    /* Weights into a CPU backend buffer */
    fprintf(stderr, "loading weights from: %s\n", model_dir);
    vox_weights_t * w = vox_weights_load(model_dir, ggml_backend_cpu_buffer_type());
    if (!w) {
        fprintf(stderr, "ERROR: vox_weights_load failed\n");
        vox_tokenizer_free(tok);
        return 1;
    }
    fprintf(stderr, "vox_weights_load OK: %d tensors, %.2f GB allocated\n",
            w->n_tensors, (double) w->n_bytes / (1024.0 * 1024.0 * 1024.0));

    /* CPU backend handle */
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "ERROR: ggml_backend_cpu_init failed\n");
        vox_weights_free(w);
        vox_tokenizer_free(tok);
        return 1;
    }

    int rc = 0;
    if (run_adapter_smoke(w, backend) != 0) {
        fprintf(stderr, "adapter smoke test FAILED\n");
        rc = 1;
    }
    if (run_decoder_smoke(w, backend) != 0) {
        fprintf(stderr, "decoder smoke test FAILED\n");
        rc = 1;
    }

    ggml_backend_free(backend);
    vox_weights_free(w);
    vox_tokenizer_free(tok);

    fprintf(stderr, "\n%s\n", rc == 0 ? "smoke test OK" : "smoke test FAILED");
    return rc;
}
