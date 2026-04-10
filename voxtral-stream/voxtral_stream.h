/*
 * voxtral_stream.h - vox_stream_t streaming inference library (Phase 2)
 *
 * One model load per session, many clips per session. The Sub-goal A
 * implementation is an offline-pipeline placeholder: vox_stream_feed buffers
 * raw samples, vox_stream_finish runs the entire pipeline (mel -> conv stem
 * -> encoder -> adapter -> decoder loop) and pushes generated token IDs to
 * an internal queue, vox_stream_get drains decoded tokens to the caller.
 *
 * Sub-goal C will replace the body of vox_stream_finish with a true
 * incremental pipeline driven by vox_stream_feed, without changing this
 * header. Consumers (test_stream, future Wyoming bridge) keep the same API.
 */

#ifndef VOXTRAL_STREAM_H
#define VOXTRAL_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vox_stream vox_stream_t;

typedef struct {
    int backend;            /* 0 = CPU, 1 = CUDA. CUDA is the default. */
    int delay_tokens;       /* sinusoidal time-embedding delay; voxtral default = 6 */
    int max_audio_seconds;  /* sizes per-clip sample buffer + decoder KV cache */
    int verbose;            /* 0 = quiet, 1 = log per-stage progress to stderr */
    int quant;              /* 0 = BF16 (default), 1 = Q8_0 */
} vox_stream_opts_t;

/* Sensible defaults: backend=cuda, delay_tokens=6, max_audio_seconds=120, quiet. */
vox_stream_opts_t vox_stream_default_opts(void);

/* Token returned by vox_stream_get. `text` is owned by the tokenizer and
 * remains valid until vox_stream_free is called. */
typedef struct {
    int          id;
    const char * text;
} vox_token_t;

/* Initialize: load tokenizer + weights, set up backend, precompute ada_scaled,
 * allocate decoder KV cache. Returns NULL on failure (reason logged to stderr). */
vox_stream_t * vox_stream_init(const char * model_dir, const vox_stream_opts_t * opts);

/* Append PCM samples (mono float32 16 kHz, range [-1, 1]). Sub-goal A buffers
 * the samples internally; no model work happens until vox_stream_finish. */
int vox_stream_feed(vox_stream_t * s, const float * samples, int n_samples);

/* Signal end-of-audio. Runs the offline pipeline (Sub-goal A) and fills the
 * token queue. Returns 0 on success, negative on error. */
int vox_stream_finish(vox_stream_t * s);

/* Drain up to `max` ready tokens into `out`. Returns the number written.
 * Returns 0 once the queue is empty. */
int vox_stream_get(vox_stream_t * s, vox_token_t * out, int max);

/* Clear per-clip state (sample buffer, token queue). Keeps the loaded model,
 * backend, and ada_scaled. Cheap. Used to drive multiple clips through one
 * vox_stream_t in the test harness. */
void vox_stream_reset(vox_stream_t * s);

/* Release everything. Safe to call on a NULL or partially-initialized stream. */
void vox_stream_free(vox_stream_t * s);

/* Recompute ada_scaled with a new delay. Cheap (one-shot CPU work). Use
 * between clips, not mid-clip. */
void vox_stream_set_delay(vox_stream_t * s, int delay_tokens);

#ifdef __cplusplus
}
#endif

#endif /* VOXTRAL_STREAM_H */
