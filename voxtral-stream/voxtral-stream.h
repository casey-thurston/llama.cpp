/*
 * voxtral-stream.h - Streaming Voxtral Realtime 4B speech-to-text on ggml.
 *
 * Public C API mirrors antirez/voxtral.c's vox_stream_* surface so a Python
 * ctypes wrapper (or C caller) sees the same shape regardless of backend.
 *
 * Phase 1: API surface only. Implementation lands in voxtral-stream.c.
 */

#ifndef VOXTRAL_STREAM_H
#define VOXTRAL_STREAM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vox_stream vox_stream_t;

/* Load model weights from `model_dir` (containing consolidated.safetensors +
 * tekken.json) and create a streaming context. delay_ms is the transcription
 * delay in milliseconds (80-2400, default 480). Returns NULL on failure. */
vox_stream_t * vox_stream_init(const char * model_dir, int delay_ms);

/* Feed mono float32 16kHz samples in [-1,1]. Runs encoder/decoder on whatever
 * is now ready and queues output tokens internally. Returns 0 on success. */
int vox_stream_feed(vox_stream_t * s, const float * samples, int n_samples);

/* Drain pending decoded token strings. Fills out_tokens with up to `max`
 * pointers into per-stream string storage. Pointers remain valid until the
 * next vox_stream_feed/finish/free call. Returns the number of tokens written
 * (0 = nothing pending). */
int vox_stream_get(vox_stream_t * s, const char ** out_tokens, int max);

/* Signal end-of-audio. Right-pads audio so the conv stem terminates cleanly,
 * then runs final encoder + decoder iterations. Returns 0 on success. */
int vox_stream_finish(vox_stream_t * s);

/* Free streaming context and all weights. */
void vox_stream_free(vox_stream_t * s);

/* Set minimum time between encoder runs, in seconds. Lower = more responsive
 * (higher GPU overhead), higher = more efficient batching. Default: 0.5. */
void vox_set_processing_interval(vox_stream_t * s, float seconds);

#ifdef __cplusplus
}
#endif

#endif /* VOXTRAL_STREAM_H */
