/*
 * test_stream.c -- streaming-paced test harness for vox_stream_t.
 *
 * Loads the model ONCE, drives multiple WAV files through the library at
 * the configured pacing rate, measures TTFA / wall / RTF per clip. This is
 * the canonical benchmark/correctness binary going forwards -- every future
 * optimization commit should be validated through this harness.
 *
 * With Sub-goal A internals (offline-pipeline placeholder), tokens only
 * become available after vox_stream_finish, so TTFA ~ wall. Sub-goal C
 * replaces the library internals with a true incremental pipeline; this
 * harness does not change between A and C.
 *
 * Usage:
 *   test_stream -m <model_dir> -i <wav> [-i <wav>...]
 *               [--pace realtime|asfast|chunk-ms N]   (default: realtime)
 *               [--delay-tokens N]                    (default: 6)
 *               [--warmup]
 *               [--csv <path>]
 *               [-v]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ggml.h"
#include "voxtral_stream.h"
#include "voxtral_audio.h"

#define MAX_CLIPS 32
#define TRANSCRIPT_BUF_SIZE 16384

typedef enum {
    PACE_REALTIME,
    PACE_ASFAST,
    PACE_CHUNK_MS,
} pace_mode_t;

typedef struct {
    pace_mode_t mode;
    int         chunk_ms;
} pace_t;

typedef struct {
    const char * clip;
    double       audio_seconds;
    double       ttfa_ms;
    double       wall_s;
    double       rtf;
    int          n_tokens;
} clip_metrics_t;

static const char * pace_name(pace_t p) {
    switch (p.mode) {
        case PACE_REALTIME: return "realtime";
        case PACE_ASFAST:   return "asfast";
        case PACE_CHUNK_MS: return "chunk-ms";
    }
    return "?";
}

static double now_s(void) {
    return ggml_time_us() * 1e-6;
}

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m <model_dir> -i <wav> [-i <wav>...]\n"
        "          [--pace realtime|asfast|chunk-ms N]   (default: realtime)\n"
        "          [--delay-tokens N]                    (default: 6)\n"
        "          [--warmup]\n"
        "          [--csv <path>]\n"
        "          [-v]\n",
        argv0);
}

/* Drain ready tokens, append to `transcript`, mark TTFA on first emit. */
static void drain_tokens(vox_stream_t * s, double t_start, clip_metrics_t * m,
                         char * transcript, int * transcript_len, int transcript_cap) {
    vox_token_t batch[64];
    for (;;) {
        int got = vox_stream_get(s, batch, 64);
        if (got <= 0) break;
        for (int i = 0; i < got; i++) {
            if (m->ttfa_ms < 0) m->ttfa_ms = (now_s() - t_start) * 1000.0;
            if (batch[i].text) {
                int slen = (int) strlen(batch[i].text);
                if (*transcript_len + slen + 1 < transcript_cap) {
                    memcpy(transcript + *transcript_len, batch[i].text, slen);
                    *transcript_len += slen;
                    transcript[*transcript_len] = 0;
                }
            }
            m->n_tokens++;
        }
    }
}

/* Drive one clip through the library. */
static int run_clip(vox_stream_t * s, const char * wav_path, pace_t pace, clip_metrics_t * m) {
    int n_samples = 0;
    float * raw = vox_load_wav(wav_path, &n_samples);
    if (!raw) { fprintf(stderr, "vox_load_wav failed: %s\n", wav_path); return -1; }
    m->clip = wav_path;
    m->audio_seconds = n_samples / 16000.0;
    m->ttfa_ms = -1.0;
    m->n_tokens = 0;
    m->wall_s = 0;
    m->rtf = 0;

    int chunk_samples;
    if (pace.mode == PACE_ASFAST) {
        chunk_samples = n_samples > 0 ? n_samples : 1;     /* one shot */
    } else {
        chunk_samples = (pace.chunk_ms * 16000) / 1000;
        if (chunk_samples <= 0) chunk_samples = 1280;       /* fall back to 80ms */
    }

    char transcript[TRANSCRIPT_BUF_SIZE];
    int  transcript_len = 0;
    transcript[0] = 0;

    double t_start = now_s();
    int offset = 0;
    while (offset < n_samples) {
        int n = chunk_samples;
        if (offset + n > n_samples) n = n_samples - offset;
        if (vox_stream_feed(s, raw + offset, n) != 0) {
            fprintf(stderr, "vox_stream_feed failed\n");
            free(raw); return -1;
        }
        offset += n;

        drain_tokens(s, t_start, m, transcript, &transcript_len, TRANSCRIPT_BUF_SIZE);

        if (pace.mode == PACE_REALTIME || pace.mode == PACE_CHUNK_MS) {
            usleep((useconds_t) pace.chunk_ms * 1000);
        }
    }

    if (vox_stream_finish(s) != 0) {
        fprintf(stderr, "vox_stream_finish failed\n");
        free(raw); return -1;
    }

    drain_tokens(s, t_start, m, transcript, &transcript_len, TRANSCRIPT_BUF_SIZE);

    double t_end = now_s();
    m->wall_s = t_end - t_start;
    m->rtf = m->audio_seconds > 0 ? m->wall_s / m->audio_seconds : 0;

    printf("[%s] %s\n", wav_path, transcript);
    fflush(stdout);

    free(raw);
    return 0;
}

int main(int argc, char ** argv) {
    const char * model_dir = NULL;
    const char * clips[MAX_CLIPS];
    int n_clips = 0;
    pace_t pace = { PACE_REALTIME, 80 };
    int delay_tokens = 6;
    int warmup = 0;
    const char * csv_path = NULL;
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (n_clips >= MAX_CLIPS) { fprintf(stderr, "too many -i clips\n"); return 2; }
            clips[n_clips++] = argv[++i];
        } else if (strcmp(argv[i], "--pace") == 0 && i + 1 < argc) {
            const char * p = argv[++i];
            if (strcmp(p, "realtime") == 0)    { pace.mode = PACE_REALTIME; pace.chunk_ms = 80; }
            else if (strcmp(p, "asfast") == 0) { pace.mode = PACE_ASFAST;   pace.chunk_ms = 0; }
            else if (strcmp(p, "chunk-ms") == 0 && i + 1 < argc) {
                pace.mode = PACE_CHUNK_MS;
                pace.chunk_ms = atoi(argv[++i]);
                if (pace.chunk_ms <= 0) { fprintf(stderr, "bad chunk-ms\n"); return 2; }
            } else { fprintf(stderr, "bad --pace value: %s\n", p); return 2; }
        } else if (strcmp(argv[i], "--delay-tokens") == 0 && i + 1 < argc) {
            delay_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0) {
            warmup = 1;
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else { usage(argv[0]); return 2; }
    }
    if (!model_dir || n_clips == 0) { usage(argv[0]); return 2; }

    /* ---- Load the model ONCE for the whole session ---- */
    vox_stream_opts_t opts = vox_stream_default_opts();
    opts.delay_tokens = delay_tokens;
    opts.verbose = verbose;
    vox_stream_t * s = vox_stream_init(model_dir, &opts);
    if (!s) return 1;

    fprintf(stderr, "test_stream: model loaded; running %d clip(s) with --pace=%s%s\n",
            n_clips, pace_name(pace), warmup ? " (with warmup)" : "");

    FILE * csv = NULL;
    if (csv_path) {
        csv = fopen(csv_path, "a");
        if (csv) {
            fseek(csv, 0, SEEK_END);
            if (ftell(csv) == 0) {
                fprintf(csv, "clip\tpace\tchunk_ms\tdelay_tokens\taudio_s\tttfa_ms\twall_s\trtf\tn_tokens\n");
            }
        }
    }

    int rc = 0;
    for (int i = 0; i < n_clips; i++) {
        clip_metrics_t m;

        if (warmup) {
            vox_stream_reset(s);
            if (run_clip(s, clips[i], pace, &m) != 0) { rc = 1; break; }
            fprintf(stderr, "  [warmup discarded]\n");
        }

        vox_stream_reset(s);
        if (run_clip(s, clips[i], pace, &m) != 0) { rc = 1; break; }

        fprintf(stderr,
                "  audio=%.2fs ttfa=%.0fms wall=%.2fs rtf=%.3f tokens=%d\n",
                m.audio_seconds, m.ttfa_ms, m.wall_s, m.rtf, m.n_tokens);

        if (csv) {
            fprintf(csv, "%s\t%s\t%d\t%d\t%.3f\t%.1f\t%.3f\t%.4f\t%d\n",
                    m.clip, pace_name(pace), pace.chunk_ms, delay_tokens,
                    m.audio_seconds, m.ttfa_ms, m.wall_s, m.rtf, m.n_tokens);
            fflush(csv);
        }
    }

    if (csv) fclose(csv);
    vox_stream_free(s);
    return rc;
}
