#include <obs-module.h>

#ifdef _WIN32
#include <fstream>
#define NOMINMAX
#include <windows.h>
#undef max
#endif

#include <string>
#include <thread>
#include <mutex>
#include <cinttypes>
#include <algorithm>
#include <regex>
#include <functional>

#include <whisper.h>

#include "cleanstream-filter.h"
#include "model-utils/model-downloader.h"
#include "whisper-utils/whisper-language.h"
#include "whisper-utils/whisper-processing.h"
#include "whisper-utils/whisper-utils.h"
#include "cleanstream-filter-data.h"

#include "plugin-support.h"

// buffer size in msec
#define BUFFER_SIZE_MSEC 1010
// at 16Khz, 1010 msec is 16160 frames
#define WHISPER_FRAME_SIZE 16160
// overlap in msec
#define OVERLAP_SIZE_MSEC 340

#define VAD_THOLD 0.0001f
#define FREQ_THOLD 100.0f

#define MT_ obs_module_text

void whisper_loop(void *data);

inline enum speaker_layout convert_speaker_layout(uint8_t channels)
{
	switch (channels) {
	case 0:
		return SPEAKERS_UNKNOWN;
	case 1:
		return SPEAKERS_MONO;
	case 2:
		return SPEAKERS_STEREO;
	case 3:
		return SPEAKERS_2POINT1;
	case 4:
		return SPEAKERS_4POINT0;
	case 5:
		return SPEAKERS_4POINT1;
	case 6:
		return SPEAKERS_5POINT1;
	case 8:
		return SPEAKERS_7POINT1;
	default:
		return SPEAKERS_UNKNOWN;
	}
}

struct obs_audio_data *cleanstream_filter_audio(void *data, struct obs_audio_data *audio)
{
	if (data == nullptr) {
		return audio;
	}

	struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);

	if (!gf->active) {
		return audio;
	}

	if (gf->whisper_context == nullptr) {
		// Whisper not initialized, just pass through
		return audio;
	}

	size_t input_buffer_size = 0;

	// std::lock_guard<std::mutex> lock(gf->whisper_buf_mutex); // scoped lock

	if (audio != nullptr && audio->frames > 0) {
		// push back current audio data to input circlebuf
		for (size_t c = 0; c < gf->channels; c++) {
			circlebuf_push_back(&gf->input_buffers[c], audio->data[c],
					    audio->frames * sizeof(float));
		}
		// push audio packet info (timestamp/frame count) to info circlebuf
		struct cleanstream_audio_info info = {0};
		info.frames = audio->frames;       // number of frames in this packet
		info.timestamp = audio->timestamp; // timestamp of this packet
		circlebuf_push_back(&gf->info_buffer, &info, sizeof(info));
	}

	// check the size of the input buffer - if it's more than 1500ms worth of audio, start playback
	if (gf->input_buffers[0].size > 1500 * gf->sample_rate * sizeof(float) / 1000) {
		// find needed number of frames from the incoming audio
		size_t num_frames_needed = audio->frames;

		gf->output_audio.timestamp = 0;

		std::vector<float> temporary_buffers[MAX_AUDIO_CHANNELS];

		while (temporary_buffers[0].size() < num_frames_needed) {
			struct cleanstream_audio_info info_out = {0};
			// pop from input buffers to get audio packet info
			circlebuf_pop_front(&gf->info_buffer, &info_out, sizeof(info_out));
			if (gf->output_audio.timestamp == 0) {
				// the first packet, set the timestamp
				gf->output_audio.timestamp = info_out.timestamp;
			}

			// pop from input circlebuf to audio data
			for (size_t i = 0; i < gf->channels; i++) {
				// increase the size of the temporary buffer to hold the incoming audio in addition
				// to the existing audio on the temporary buffer
				temporary_buffers[i].resize(temporary_buffers[i].size() +
							    info_out.frames);
				circlebuf_pop_front(&gf->input_buffers[i],
						    temporary_buffers[i].data() +
							    temporary_buffers[i].size() -
							    info_out.frames,
						    info_out.frames * sizeof(float));
			}
		}
		const size_t num_frames = temporary_buffers[0].size();
		// prepare input data buffer
		da_resize(gf->output_data, num_frames * sizeof(float) * gf->channels);
		for (size_t i = 0; i < gf->channels; i++) {
			memcpy(gf->output_data.array + i * num_frames, temporary_buffers[i].data(),
			       num_frames * sizeof(float));
			gf->output_audio.data[i] =
				(uint8_t *)&gf->output_data.array[i * num_frames];
		}

		if (gf->current_result == DetectionResult::DETECTION_RESULT_FILLER ||
		    gf->current_result == DetectionResult::DETECTION_RESULT_BEEP) {
			obs_log(LOG_INFO, "Filler detected");
			// set the audio to 0
			for (size_t i = 0; i < gf->channels; i++) {
				memset(gf->output_audio.data[i], 0, num_frames * sizeof(float));
			}
		}

		gf->output_audio.frames = (uint32_t)num_frames;

		return &gf->output_audio;
	}

	return NULL;
}

const char *cleanstream_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return MT_("CleanStreamAudioFilter");
}

void cleanstream_destroy(void *data)
{
	struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);

	obs_log(LOG_INFO, "cleanstream_destroy");
	{
		std::lock_guard<std::mutex> lock(gf->whisper_ctx_mutex);
		if (gf->whisper_context != nullptr) {
			whisper_free(gf->whisper_context);
			gf->whisper_context = nullptr;
		}
	}
	// join the thread
	gf->whisper_thread.join();

	if (gf->resampler) {
		audio_resampler_destroy(gf->resampler);
		audio_resampler_destroy(gf->resampler_back);
	}
	{
		std::lock_guard<std::mutex> lockbuf(gf->whisper_buf_mutex);
		std::lock_guard<std::mutex> lockoutbuf(gf->whisper_outbuf_mutex);
		bfree(gf->copy_buffers[0]);
		gf->copy_buffers[0] = nullptr;
		for (size_t i = 0; i < gf->channels; i++) {
			circlebuf_free(&gf->input_buffers[i]);
		}
	}
	circlebuf_free(&gf->info_buffer);
	da_free(gf->output_data);

	bfree(gf);
}

void cleanstream_update(void *data, obs_data_t *s)
{
	obs_log(LOG_INFO, "cleanstream_update");

	struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);

	gf->filler_p_threshold = (float)obs_data_get_double(s, "filler_p_threshold");
	gf->log_level = (int)obs_data_get_int(s, "log_level");
	gf->do_silence = obs_data_get_bool(s, "do_silence");
	gf->vad_enabled = obs_data_get_bool(s, "vad_enabled");
	gf->detect_regex = obs_data_get_string(s, "detect_regex");
	gf->beep_regex = obs_data_get_string(s, "beep_regex");
	gf->log_words = obs_data_get_bool(s, "log_words");

	obs_log(gf->log_level, "update whisper model");
	update_whisper_model(gf, s);

	{
		std::lock_guard<std::mutex> lock(gf->whisper_ctx_mutex);

		gf->whisper_params = whisper_full_default_params(
			(whisper_sampling_strategy)obs_data_get_int(s, "whisper_sampling_method"));
		gf->whisper_params.duration_ms = BUFFER_SIZE_MSEC;
		gf->whisper_params.language = obs_data_get_string(s, "whisper_language_select");
		gf->whisper_params.translate = false;
		gf->whisper_params.initial_prompt = obs_data_get_string(s, "initial_prompt");
		gf->whisper_params.n_threads = (int)obs_data_get_int(s, "n_threads");
		gf->whisper_params.n_max_text_ctx = (int)obs_data_get_int(s, "n_max_text_ctx");
		gf->whisper_params.no_context = obs_data_get_bool(s, "no_context");
		gf->whisper_params.single_segment = obs_data_get_bool(s, "single_segment");
		gf->whisper_params.print_special = obs_data_get_bool(s, "print_special");
		gf->whisper_params.print_progress = obs_data_get_bool(s, "print_progress");
		gf->whisper_params.print_realtime = obs_data_get_bool(s, "print_realtime");
		gf->whisper_params.print_timestamps = obs_data_get_bool(s, "print_timestamps");
		gf->whisper_params.token_timestamps = obs_data_get_bool(s, "token_timestamps");
		gf->whisper_params.thold_pt = (float)obs_data_get_double(s, "thold_pt");
		gf->whisper_params.thold_ptsum = (float)obs_data_get_double(s, "thold_ptsum");
		gf->whisper_params.max_len = (int)obs_data_get_int(s, "max_len");
		gf->whisper_params.split_on_word = obs_data_get_bool(s, "split_on_word");
		gf->whisper_params.max_tokens = (int)obs_data_get_int(s, "max_tokens");
		gf->whisper_params.speed_up = obs_data_get_bool(s, "speed_up");
		gf->whisper_params.suppress_blank = obs_data_get_bool(s, "suppress_blank");
		gf->whisper_params.suppress_non_speech_tokens =
			obs_data_get_bool(s, "suppress_non_speech_tokens");
		gf->whisper_params.temperature = (float)obs_data_get_double(s, "temperature");
		gf->whisper_params.max_initial_ts = (float)obs_data_get_double(s, "max_initial_ts");
		gf->whisper_params.length_penalty = (float)obs_data_get_double(s, "length_penalty");
	}

	obs_log(LOG_INFO, "cleanstream update finished");
}

void *cleanstream_create(obs_data_t *settings, obs_source_t *filter)
{
	obs_log(LOG_INFO, "cleanstream create");

	void *data = bmalloc(sizeof(struct cleanstream_data));
	struct cleanstream_data *gf = new (data) cleanstream_data();

	// Get the number of channels for the input source
	gf->channels = audio_output_get_channels(obs_get_audio());
	gf->sample_rate = audio_output_get_sample_rate(obs_get_audio());
	gf->frames = (size_t)((float)gf->sample_rate / (1000.0f / (float)BUFFER_SIZE_MSEC));
	gf->last_num_frames = 0;

	for (size_t i = 0; i < MAX_AUDIO_CHANNELS; i++) {
		circlebuf_init(&gf->input_buffers[i]);
		gf->output_audio.data[i] = nullptr;
	}
	circlebuf_init(&gf->info_buffer);
	da_init(gf->output_data);

	gf->output_audio.frames = 0;
	gf->output_audio.timestamp = 0;

	// allocate copy buffers
	gf->copy_buffers[0] =
		static_cast<float *>(bzalloc(gf->channels * gf->frames * sizeof(float)));
	for (size_t c = 1; c < gf->channels; c++) { // set the channel pointers
		gf->copy_buffers[c] = gf->copy_buffers[0] + c * gf->frames;
	}

	gf->context = filter;
	gf->whisper_model_path = std::string(""); // The update function will set the model path
	gf->whisper_context = nullptr;

	gf->overlap_ms = OVERLAP_SIZE_MSEC;
	gf->overlap_frames = (size_t)((float)gf->sample_rate / (1000.0f / (float)gf->overlap_ms));
	obs_log(LOG_INFO, "CleanStream filter: channels %d, frames %d, sample_rate %d",
		(int)gf->channels, (int)gf->frames, gf->sample_rate);

	struct resample_info src, dst;
	src.samples_per_sec = gf->sample_rate;
	src.format = AUDIO_FORMAT_FLOAT_PLANAR;
	src.speakers = convert_speaker_layout((uint8_t)gf->channels);

	dst.samples_per_sec = WHISPER_SAMPLE_RATE;
	dst.format = AUDIO_FORMAT_FLOAT_PLANAR;
	dst.speakers = convert_speaker_layout((uint8_t)1);

	gf->resampler = audio_resampler_create(&dst, &src);
	gf->resampler_back = audio_resampler_create(&src, &dst);

	gf->active = true;
	gf->detect_regex = nullptr;
	gf->beep_regex = nullptr;

	// call the update function to set the whisper model
	cleanstream_update(gf, settings);

	return gf;
}

void cleanstream_activate(void *data)
{
	struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);
	obs_log(LOG_INFO, "CleanStream filter activated");
	gf->active = true;
}

void cleanstream_deactivate(void *data)
{
	struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);
	obs_log(LOG_INFO, "CleanStream filter deactivated");
	gf->active = false;
}

void cleanstream_defaults(obs_data_t *s)
{
	obs_data_set_default_double(s, "filler_p_threshold", 0.75);
	obs_data_set_default_bool(s, "do_silence", true);
	obs_data_set_default_bool(s, "vad_enabled", true);
	obs_data_set_default_int(s, "log_level", LOG_INFO);
	obs_data_set_default_string(s, "detect_regex", "\\b(uh+|um+)\\.?(\\.{2,})?\\b");
	// Profane words taken from https://en.wiktionary.org/wiki/Category:English_swear_words
	obs_data_set_default_string(
		s, "beep_regex",
		"(fuck)|(shit)|(bitch)|(cunt)|(pussy)|(dick)|(asshole)|(whore)|(cock)|(nigger)|(nigga)|(prick)");
	obs_data_set_default_bool(s, "log_words", true);
	obs_data_set_default_string(s, "whisper_model_path", "Whisper Tiny English (74Mb)");
	obs_data_set_default_string(s, "whisper_language_select", "en");

	// Whisper parameters
	obs_data_set_default_int(s, "whisper_sampling_method", WHISPER_SAMPLING_BEAM_SEARCH);
	obs_data_set_default_string(s, "initial_prompt", "uhm, Uh, um, Uhh, um. um... uh. uh... ");
	obs_data_set_default_int(s, "n_threads", 4);
	obs_data_set_default_int(s, "n_max_text_ctx", 16384);
	obs_data_set_default_bool(s, "no_context", true);
	obs_data_set_default_bool(s, "single_segment", true);
	obs_data_set_default_bool(s, "print_special", false);
	obs_data_set_default_bool(s, "print_progress", false);
	obs_data_set_default_bool(s, "print_realtime", false);
	obs_data_set_default_bool(s, "print_timestamps", false);
	obs_data_set_default_bool(s, "token_timestamps", false);
	obs_data_set_default_double(s, "thold_pt", 0.01);
	obs_data_set_default_double(s, "thold_ptsum", 0.01);
	obs_data_set_default_int(s, "max_len", 0);
	obs_data_set_default_bool(s, "split_on_word", false);
	obs_data_set_default_int(s, "max_tokens", 3);
	obs_data_set_default_bool(s, "speed_up", false);
	obs_data_set_default_bool(s, "suppress_blank", true);
	obs_data_set_default_bool(s, "suppress_non_speech_tokens", true);
	obs_data_set_default_double(s, "temperature", 0.5);
	obs_data_set_default_double(s, "max_initial_ts", 1.0);
	obs_data_set_default_double(s, "length_penalty", -1.0);
}

obs_properties_t *cleanstream_properties(void *data)
{
	obs_properties_t *ppts = obs_properties_create();

	obs_properties_add_float_slider(ppts, "filler_p_threshold", "filler_p_threshold", 0.0f,
					1.0f, 0.05f);
	obs_properties_add_bool(ppts, "do_silence", "do_silence");
	obs_properties_add_bool(ppts, "vad_enabled", "vad_enabled");
	obs_property_t *list = obs_properties_add_list(ppts, "log_level", "log_level",
						       OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(list, "DEBUG", LOG_DEBUG);
	obs_property_list_add_int(list, "INFO", LOG_INFO);
	obs_property_list_add_int(list, "WARNING", LOG_WARNING);
	obs_properties_add_bool(ppts, "log_words", "log_words");
	obs_properties_add_text(ppts, "detect_regex", "detect_regex", OBS_TEXT_DEFAULT);
	obs_properties_add_text(ppts, "beep_regex", "beep_regex", OBS_TEXT_DEFAULT);

	// Add a list of available whisper models to download
	obs_property_t *whisper_models_list =
		obs_properties_add_list(ppts, "whisper_model_path", MT_("whisper_model"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	// Add models from models_info map
	for (const auto &model_info : models_info) {
		if (model_info.second.type == MODEL_TYPE_TRANSCRIPTION) {
			obs_property_list_add_string(whisper_models_list, model_info.first.c_str(),
						     model_info.first.c_str());
		}
	}

	obs_properties_t *whisper_params_group = obs_properties_create();
	obs_properties_add_group(ppts, "whisper_params_group", "Whisper Parameters",
				 OBS_GROUP_NORMAL, whisper_params_group);

	// Add language selector
	obs_property_t *whisper_language_select_list =
		obs_properties_add_list(whisper_params_group, "whisper_language_select", "Language",
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	// iterate over all available languages in whisper_available_lang map<string, string>
	for (auto const &pair : whisper_available_lang) {
		obs_property_list_add_string(whisper_language_select_list, pair.second.c_str(),
					     pair.first.c_str());
	}

	obs_property_t *whisper_sampling_method_list = obs_properties_add_list(
		whisper_params_group, "whisper_sampling_method", "whisper_sampling_method",
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(whisper_sampling_method_list, "Beam search",
				  WHISPER_SAMPLING_BEAM_SEARCH);
	obs_property_list_add_int(whisper_sampling_method_list, "Greedy", WHISPER_SAMPLING_GREEDY);

	// int n_threads;
	obs_properties_add_int_slider(whisper_params_group, "n_threads", "n_threads", 1, 8, 1);
	// int n_max_text_ctx;     // max tokens to use from past text as prompt for the decoder
	obs_properties_add_int_slider(whisper_params_group, "n_max_text_ctx", "n_max_text_ctx", 0,
				      16384, 100);
	// int offset_ms;          // start offset in ms
	// int duration_ms;        // audio duration to process in ms
	// bool translate;
	// bool no_context;        // do not use past transcription (if any) as initial prompt for the decoder
	obs_properties_add_bool(whisper_params_group, "no_context", "no_context");
	// bool single_segment;    // force single segment output (useful for streaming)
	obs_properties_add_bool(whisper_params_group, "single_segment", "single_segment");
	// bool print_special;     // print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)
	obs_properties_add_bool(whisper_params_group, "print_special", "print_special");
	// bool print_progress;    // print progress information
	obs_properties_add_bool(whisper_params_group, "print_progress", "print_progress");
	// bool print_realtime;    // print results from within whisper.cpp (avoid it, use callback instead)
	obs_properties_add_bool(whisper_params_group, "print_realtime", "print_realtime");
	// bool print_timestamps;  // print timestamps for each text segment when printing realtime
	obs_properties_add_bool(whisper_params_group, "print_timestamps", "print_timestamps");
	// bool  token_timestamps; // enable token-level timestamps
	obs_properties_add_bool(whisper_params_group, "token_timestamps", "token_timestamps");
	// float thold_pt;         // timestamp token probability threshold (~0.01)
	obs_properties_add_float_slider(whisper_params_group, "thold_pt", "thold_pt", 0.0f, 1.0f,
					0.05f);
	// float thold_ptsum;      // timestamp token sum probability threshold (~0.01)
	obs_properties_add_float_slider(whisper_params_group, "thold_ptsum", "thold_ptsum", 0.0f,
					1.0f, 0.05f);
	// int   max_len;          // max segment length in characters
	obs_properties_add_int_slider(whisper_params_group, "max_len", "max_len", 0, 100, 1);
	// bool  split_on_word;    // split on word rather than on token (when used with max_len)
	obs_properties_add_bool(whisper_params_group, "split_on_word", "split_on_word");
	// int   max_tokens;       // max tokens per segment (0 = no limit)
	obs_properties_add_int_slider(whisper_params_group, "max_tokens", "max_tokens", 0, 100, 1);
	// bool speed_up;          // speed-up the audio by 2x using Phase Vocoder
	obs_properties_add_bool(whisper_params_group, "speed_up", "speed_up");
	// const char * initial_prompt;
	obs_properties_add_text(whisper_params_group, "initial_prompt", "initial_prompt",
				OBS_TEXT_DEFAULT);
	// bool suppress_blank
	obs_properties_add_bool(whisper_params_group, "suppress_blank", "suppress_blank");
	// bool suppress_non_speech_tokens
	obs_properties_add_bool(whisper_params_group, "suppress_non_speech_tokens",
				"suppress_non_speech_tokens");
	// float temperature
	obs_properties_add_float_slider(whisper_params_group, "temperature", "temperature", 0.0f,
					1.0f, 0.05f);
	// float max_initial_ts
	obs_properties_add_float_slider(whisper_params_group, "max_initial_ts", "max_initial_ts",
					0.0f, 1.0f, 0.05f);
	// float length_penalty
	obs_properties_add_float_slider(whisper_params_group, "length_penalty", "length_penalty",
					-1.0f, 1.0f, 0.1f);

	UNUSED_PARAMETER(data);
	return ppts;
}
