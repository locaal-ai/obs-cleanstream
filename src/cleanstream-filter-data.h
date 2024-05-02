#ifndef CLEANSTREAM_FILTER_DATA_H
#define CLEANSTREAM_FILTER_DATA_H

#include <obs.h>
#include <string>
#include <thread>
#include <mutex>

#include <util/circlebuf.h>
#include <util/darray.h>
#include <media-io/audio-resampler.h>

#include "whisper-utils/silero-vad-onnx.h"

#include <whisper.h>

#define MAX_PREPROC_CHANNELS 2

enum ReplaceSounds {
	REPLACE_SOUNDS_NONE = 0,
	REPLACE_SOUNDS_BEEP = 1,
	REPLACE_SOUNDS_SILENCE = 2,
	REPLACE_SOUNDS_HORN = 3,
	REPLACE_SOUNDS_EXTERNAL = 4,
};

// Audio packet info
struct cleanstream_audio_info {
	uint32_t frames;
	uint64_t timestamp;
};

struct cleanstream_data {
	obs_source_t *context; // obs input source
	size_t channels;       // number of channels
	uint32_t sample_rate;  // input sample rate
	// How many input frames (in input sample rate) are needed for the next whisper frame
	size_t frames;
	// How many ms/frames are needed to overlap with the next whisper frame
	size_t overlap_frames;
	size_t overlap_ms;
	// How many frames were processed in the last whisper frame (this is dynamic)
	size_t last_num_frames;
	int current_result;

	/* Silero VAD */
	std::unique_ptr<VadIterator> vad;

	/* PCM buffers */
	float *copy_buffers[MAX_PREPROC_CHANNELS];
	struct circlebuf info_buffer;
	struct circlebuf input_buffers[MAX_PREPROC_CHANNELS];

	/* Resampler */
	audio_resampler_t *resampler;
	audio_resampler_t *resampler_back;

	/* whisper */
	std::string whisper_model_path = "models/ggml-tiny.en.bin";
	std::string whisper_model_file_currently_loaded;
	struct whisper_context *whisper_context;
	whisper_full_params whisper_params;

	// Use std for thread and mutex
	std::thread whisper_thread;
	std::mutex whisper_buf_mutex;
	std::mutex whisper_outbuf_mutex;
	std::mutex whisper_ctx_mutex;

	/* output data */
	struct obs_audio_data output_audio;
	DARRAY(float) output_data;

	float filler_p_threshold;

	bool do_silence;
	bool vad_enabled;
	int log_level;
	const char *detect_regex;
	bool log_words;
	bool active;
	long long replace_sound;
};

#endif
