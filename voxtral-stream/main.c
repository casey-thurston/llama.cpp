/*
 * voxtral-stream main.c - Phase 1C smoke test.
 *
 * Loads all 711 Voxtral Realtime 4B weights into a single ggml CPU backend
 * buffer via vox_weights_load(), then dumps:
 *   - tokenizer vocab size and a few decoded token IDs
 *   - vox_weights_load summary (count + bytes)
 *   - shapes of the same probe tensors that the 1B smoke test printed
 *   - 8 sample F32 values from layers.0.attention_norm so they can be
 *     cross-checked against voxtral.c.
 *
 * Usage:
 *   voxtral-stream <model_dir>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-backend.h"

#include "voxtral_tokenizer.h"
#include "voxtral_audio.h"
#include "voxtral_weights.h"

/* The vendored modules extern-declare these globals; voxtral.c's main module
 * defines them. We're the new "main module" so we own them here. */
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

    /* ---- Tokenizer load ---- */
    snprintf(path, sizeof(path), "%s/tekken.json", model_dir);
    fprintf(stderr, "loading tokenizer: %s\n", path);
    vox_tokenizer_t * tok = vox_tokenizer_load(path);
    if (!tok) {
        fprintf(stderr, "ERROR: failed to load tokenizer\n");
        return 1;
    }
    int vocab = vox_tokenizer_vocab_size(tok);
    int bos = vox_tokenizer_bos(tok);
    int eos = vox_tokenizer_eos(tok);
    fprintf(stderr, "tokenizer OK: vocab=%d bos=%d eos=%d\n", vocab, bos, eos);

    /* ---- Weights load (CPU backend buffer) ---- */
    fprintf(stderr, "loading weights from: %s\n", model_dir);
    vox_weights_t * w = vox_weights_load(model_dir, ggml_backend_cpu_buffer_type());
    if (!w) {
        fprintf(stderr, "ERROR: vox_weights_load failed\n");
        vox_tokenizer_free(tok);
        return 1;
    }
    fprintf(stderr, "vox_weights_load OK: %d tensors, %.2f GB allocated\n",
            w->n_tensors, (double) w->n_bytes / (1024.0 * 1024.0 * 1024.0));

    /* ---- Spot-check shapes via the populated struct ---- */
    print_shape("decoder.tok_embeddings", w->decoder.tok_embeddings);
    print_shape("encoder.conv0", w->encoder.conv0);
    print_shape("encoder.layers[0].wq", w->encoder.layers[0].wq);
    print_shape("adapter.linear0", w->adapter.linear0);
    print_shape("decoder.layers[0].wq", w->decoder.layers[0].wq);
    print_shape("decoder.layers[0].ada_norm_down", w->decoder.layers[0].ada_norm_down);
    print_shape("decoder.layers[0].ada_norm_up", w->decoder.layers[0].ada_norm_up);
    print_shape("decoder.layers[25].w2", w->decoder.layers[25].w2);
    print_shape("decoder.norm", w->decoder.norm);

    /* ---- Sample F32 values from layers.0.attention_norm ---- */
    {
        struct ggml_tensor * t = w->decoder.layers[0].attention_norm;
        if (t && t->type == GGML_TYPE_F32) {
            int64_t n = ggml_nelements(t);
            int n_print = n < 8 ? (int) n : 8;
            float buf[8] = {0};
            ggml_backend_tensor_get(t, buf, 0, sizeof(float) * n_print);
            fprintf(stderr, "  decoder.layers[0].attention_norm[0..%d] =", n_print - 1);
            for (int i = 0; i < n_print; i++) fprintf(stderr, " %.6f", buf[i]);
            fprintf(stderr, "\n");
        }
    }

    vox_weights_free(w);
    vox_tokenizer_free(tok);

    fprintf(stderr, "smoke test OK\n");
    return 0;
}
