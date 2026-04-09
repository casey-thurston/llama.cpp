/*
 * voxtral-stream main.c - Phase 1B smoke test.
 *
 * Verifies that the vendored voxtral.c modules (tokenizer + safetensors +
 * audio) build and run correctly inside our llama.cpp/voxtral-stream tree
 * before we wire any ggml graphs to them.
 *
 * Usage:
 *   voxtral-stream <model_dir>
 *
 * Where <model_dir> contains:
 *   - consolidated.safetensors  (8.86 GB BF16, 711 tensors)
 *   - tekken.json               (Tekken tokenizer vocab)
 *
 * The smoke test prints:
 *   - tokenizer vocab size and a few decoded token IDs
 *   - first/last few safetensor tensor names + shapes
 *
 * The real CLI (mirroring voxtral.c's main.c flags: -m / --backend / -i /
 * --stdin) lands in 1E.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"

/* voxtral-stream.h declares the public API but the implementation lands in
 * 1E (voxtral-stream.c). Including it now would create unresolved-symbol
 * link errors, so we only include the vendored modules used by the smoke
 * test for the moment. */
#include "voxtral_tokenizer.h"
#include "voxtral_audio.h"
#include "voxtral_safetensors.h"

/* The vendored modules extern-declare these globals; voxtral.c's main module
 * defines them. We're the new "main module" so we own them here. */
int vox_verbose = 0;

static void print_usage(const char * argv0) {
    fprintf(stderr, "usage: %s <model_dir>\n", argv0);
    fprintf(stderr, "  <model_dir> must contain consolidated.safetensors and tekken.json\n");
}

int main(int argc, char ** argv) {
    /* Always-on touch of ggml so the linker pulls in libggml — keeps the
     * Phase 1A scaffold check intact. */
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
    /* Spot-check a few token decodings. */
    const int probe_ids[] = { 1, 2, 32, 1000, 1001, 1500, 50000 };
    for (size_t i = 0; i < sizeof(probe_ids) / sizeof(probe_ids[0]); i++) {
        const char * s = vox_tokenizer_decode(tok, probe_ids[i]);
        fprintf(stderr, "  decode(%5d) = %s\n", probe_ids[i], s ? s : "<null>");
    }

    /* ---- Safetensors load ---- */
    snprintf(path, sizeof(path), "%s/consolidated.safetensors", model_dir);
    fprintf(stderr, "loading weights: %s\n", path);
    safetensors_file_t * sf = safetensors_open(path);
    if (!sf) {
        fprintf(stderr, "ERROR: failed to mmap safetensors\n");
        vox_tokenizer_free(tok);
        return 1;
    }
    fprintf(stderr, "safetensors OK: %d tensors, %.2f GB file\n",
            sf->num_tensors, (double) sf->file_size / (1024.0 * 1024.0 * 1024.0));

    /* Spot-check a handful of expected tensor names from MODEL.md. */
    const char * probe_names[] = {
        "mm_streams_embeddings.embedding_module.tok_embeddings.weight",
        "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.weight",
        "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.layers.0.attention.wq.weight",
        "mm_streams_embeddings.embedding_module.audio_language_projection.0.weight",
        "layers.0.attention.wq.weight",
        "layers.0.ada_rms_norm_t_cond.0.weight",
        "layers.0.ada_rms_norm_t_cond.2.weight",
        "layers.25.feed_forward.w2.weight",
        "norm.weight",
    };
    for (size_t i = 0; i < sizeof(probe_names) / sizeof(probe_names[0]); i++) {
        const safetensor_t * t = safetensors_find(sf, probe_names[i]);
        if (!t) {
            fprintf(stderr, "  [MISSING] %s\n", probe_names[i]);
            continue;
        }
        fprintf(stderr, "  [%s] %s shape=[",
                t->dtype == DTYPE_BF16 ? "bf16" :
                t->dtype == DTYPE_F16 ? "f16 " :
                t->dtype == DTYPE_F32 ? "f32 " : "????",
                probe_names[i]);
        for (int d = 0; d < t->ndim; d++) {
            fprintf(stderr, "%s%lld", d ? "," : "", (long long) t->shape[d]);
        }
        fprintf(stderr, "]\n");
    }

    safetensors_close(sf);
    vox_tokenizer_free(tok);

    fprintf(stderr, "smoke test OK\n");
    return 0;
}
