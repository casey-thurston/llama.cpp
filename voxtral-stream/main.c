/*
 * voxtral-stream main.c -- thin offline CLI on top of the vox_stream_t library.
 *
 * Loads a single WAV, feeds it through vox_stream_*, prints the transcript.
 * The streaming test harness lives in test_stream.c (Sub-goal B).
 *
 * Usage: voxtral-stream -m <model_dir> -i <audio.wav>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "voxtral_stream.h"
#include "voxtral_audio.h"

static void usage(const char * argv0) {
    fprintf(stderr, "usage: %s -m <model_dir> -i <audio.wav>\n", argv0);
}

int main(int argc, char ** argv) {
    const char * model_dir = NULL;
    const char * wav_path  = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)      model_dir = argv[++i];
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) wav_path  = argv[++i];
        else { usage(argv[0]); return 2; }
    }
    if (!model_dir || !wav_path) { usage(argv[0]); return 2; }

    vox_stream_opts_t opts = vox_stream_default_opts();
    opts.verbose = 1;       /* main.c is the diagnostic CLI; always verbose */

    vox_stream_t * s = vox_stream_init(model_dir, &opts);
    if (!s) return 1;

    int n_samples = 0;
    float * raw = vox_load_wav(wav_path, &n_samples);
    if (!raw) {
        fprintf(stderr, "vox_load_wav failed: %s\n", wav_path);
        vox_stream_free(s);
        return 1;
    }
    fprintf(stderr, "audio: %d samples (%.2f s)\n", n_samples, n_samples / 16000.0);

    if (vox_stream_feed(s, raw, n_samples) != 0) {
        fprintf(stderr, "vox_stream_feed failed\n");
        free(raw);
        vox_stream_free(s);
        return 1;
    }
    free(raw);

    if (vox_stream_finish(s) != 0) {
        fprintf(stderr, "vox_stream_finish failed\n");
        vox_stream_free(s);
        return 1;
    }

    /* Drain all tokens and print as one transcript line. */
    vox_token_t batch[64];
    int total = 0;
    for (;;) {
        int got = vox_stream_get(s, batch, 64);
        if (got <= 0) break;
        for (int i = 0; i < got; i++) {
            if (batch[i].text) printf("%s", batch[i].text);
        }
        total += got;
    }
    printf("\n");
    fprintf(stderr, "\ngenerated %d tokens\n", total);

    vox_stream_free(s);
    return 0;
}
