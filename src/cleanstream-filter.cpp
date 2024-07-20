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
#include <filesystem>

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
// initial delay in msec
#define INITIAL_DELAY_MSEC 500

#define VAD_THOLD 0.0001f
#define FREQ_THOLD 100.0f

#define MT_ obs_module_text

void whisper_loop(void *data);

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
	{
		std::lock_guard<std::mutex> lock(gf->whisper_buf_mutex); // scoped lock

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
		input_buffer_size = gf->input_buffers[0].size;
	}

	// check the size of the input buffer - if it's more than <delay>ms worth of audio, start playback
	if (input_buffer_size > gf->delay_ms * gf->sample_rate * sizeof(float) / 1000) {
		// find needed number of frames from the incoming audio
		size_t num_frames_needed = audio->frames;

		std::vector<float> temporary_buffers[MAX_AUDIO_CHANNELS];
		uint64_t timestamp = 0;

		{
			std::lock_guard<std::mutex> lock(gf->whisper_buf_mutex);
			// pop from input buffers to get audio packet info
			while (temporary_buffers[0].size() < num_frames_needed) {
				struct cleanstream_audio_info info_out = {0};
				// pop from input buffers to get audio packet info
				circlebuf_pop_front(&gf->info_buffer, &info_out, sizeof(info_out));
				if (timestamp == 0) {
					timestamp = info_out.timestamp;
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
		}
		const size_t num_frames = temporary_buffers[0].size();
		const size_t frames_size_bytes = num_frames * sizeof(float);

		// prepare output data buffer
		da_resize(gf->output_data, frames_size_bytes * gf->channels);
		memset(gf->output_data.array, 0, frames_size_bytes * gf->channels);

		int inference_result = DetectionResult::DETECTION_RESULT_UNKNOWN;
		uint64_t inference_result_start_timestamp = 0;
		uint64_t inference_result_end_timestamp = 0;
		{
			std::lock_guard<std::mutex> outbuf_lock(gf->whisper_outbuf_mutex);
			inference_result = gf->current_result;
			inference_result_start_timestamp = gf->current_result_start_timestamp;
			inference_result_end_timestamp = gf->current_result_end_timestamp;
		}

		if (timestamp > inference_result_start_timestamp &&
		    timestamp < inference_result_end_timestamp) {
			if (gf->replace_sound == REPLACE_SOUNDS_SILENCE) {
				// set the audio to 0
				for (size_t i = 0; i < gf->channels; i++) {
					temporary_buffers[i].clear();
					temporary_buffers[i].resize(num_frames, 0.0f);
				}
			} else if (gf->replace_sound == REPLACE_SOUNDS_HORN ||
				   gf->replace_sound == REPLACE_SOUNDS_BEEP ||
				   gf->replace_sound == REPLACE_SOUNDS_EXTERNAL) {

				std::string replace_audio_name =
					gf->replace_sound == REPLACE_SOUNDS_HORN   ? "horn.wav"
					: gf->replace_sound == REPLACE_SOUNDS_BEEP ? "beep.wav"
					: gf->replace_sound == REPLACE_SOUNDS_EXTERNAL
						? gf->replace_sound_external
						: "";

				if (replace_audio_name != "") {
					// replace the audio with beep or horn sound
					const AudioDataFloat &replace_audio =
						gf->audioFileCache[replace_audio_name];
					if ((gf->audioFilePointer + num_frames) >=
					    replace_audio.size()) {
						gf->audioFilePointer = 0;
					}
					for (size_t i = 0; i < gf->channels; i++) {
						temporary_buffers[i].clear();
						temporary_buffers[i].insert(
							temporary_buffers[i].end(),
							replace_audio.begin() +
								gf->audioFilePointer,
							replace_audio.begin() +
								gf->audioFilePointer + num_frames);
					}
					gf->audioFilePointer += num_frames;
				}
			}
		} else {
			gf->audioFilePointer = 0;
		}

		for (size_t i = 0; i < gf->channels; i++) {
			memcpy(gf->output_data.array + i * num_frames, temporary_buffers[i].data(),
			       frames_size_bytes);
			gf->output_audio.data[i] =
				(uint8_t *)&gf->output_data.array[i * num_frames];
		}

		gf->output_audio.frames = (uint32_t)num_frames;
		gf->output_audio.timestamp = audio->timestamp;

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
	obs_log(LOG_INFO, "cleanstream_destroy");
	struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);

	gf->active = false;

	shutdown_whisper_thread(gf);

	if (gf->resampler) {
		audio_resampler_destroy(gf->resampler);
	}

	{
		std::lock_guard<std::mutex> lockbuf(gf->whisper_buf_mutex);
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

	gf->detect_regex = obs_data_get_string(s, "detect_regex");
	gf->replace_sound = obs_data_get_int(s, "replace_sound");
	gf->log_level = (int)obs_data_get_int(s, "log_level");
	gf->vad_enabled = obs_data_get_bool(s, "vad_enabled");
	gf->log_words = obs_data_get_bool(s, "log_words");
	gf->delay_ms = BUFFER_SIZE_MSEC + INITIAL_DELAY_MSEC;
	gf->current_result = DetectionResult::DETECTION_RESULT_UNKNOWN;
	gf->current_result_start_timestamp = 0;
	gf->current_result_end_timestamp = 0;

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
	gf->delay_ms = BUFFER_SIZE_MSEC + INITIAL_DELAY_MSEC;
	gf->current_result = DetectionResult::DETECTION_RESULT_UNKNOWN;
	gf->current_result_start_timestamp = 0;
	gf->current_result_end_timestamp = 0;

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

	obs_log(LOG_INFO, "CleanStream filter: channels %d, sample_rate %d", (int)gf->channels,
		gf->sample_rate);

	struct resample_info src, dst;
	src.samples_per_sec = gf->sample_rate;
	src.format = AUDIO_FORMAT_FLOAT_PLANAR;
	src.speakers = convert_speaker_layout((uint8_t)gf->channels);

	dst.samples_per_sec = WHISPER_SAMPLE_RATE;
	dst.format = AUDIO_FORMAT_FLOAT_PLANAR;
	dst.speakers = convert_speaker_layout((uint8_t)1);

	gf->resampler = audio_resampler_create(&dst, &src);

	gf->active = true;
	gf->detect_regex = nullptr;
	gf->replace_sound = REPLACE_SOUNDS_SILENCE;
	gf->replace_sound_external = "";

	// get absolute path of the audio files
	char *module_data_sounds_folder_path = obs_module_file("sounds");
	std::filesystem::path sounds_folder_path =
		std::filesystem::absolute(module_data_sounds_folder_path);
	bfree(module_data_sounds_folder_path);

#if defined(_WIN32) || defined(__APPLE__)
	// load audio files to cache
	for (const auto &file_name : {"beep.wav", "horn.wav"}) {
		std::filesystem::path audio_file_path_fs = sounds_folder_path / file_name;
		obs_log(LOG_INFO, "Loading audio file: %s", audio_file_path_fs.string().c_str());
		AudioDataFloat audioFile =
			read_audio_file(audio_file_path_fs.string().c_str(), gf->sample_rate);
		obs_log(LOG_INFO, "Loaded %lu frames of audio data", audioFile.size());
		if (audioFile.empty()) {
			obs_log(LOG_ERROR, "Failed to load audio file: %s",
				audio_file_path_fs.string().c_str());
			gf->audioFileCache[file_name] = {};
		} else {
			gf->audioFileCache[file_name] = audioFile;
		}
	}
#endif

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
	// Profane words taken from https://en.wiktionary.org/wiki/Category:English_swear_words
	obs_data_set_default_string(
		s, "detect_regex",
		"(fuck)|(shit)|(bitch)|(cunt)|(pussy)|(dick)|(asshole)|(whore)|(cock)|(nigger)|(nigga)|(prick)");
	obs_data_set_default_int(s, "replace_sound", REPLACE_SOUNDS_SILENCE);
	obs_data_set_default_bool(s, "advanced_settings", false);
	obs_data_set_default_bool(s, "vad_enabled", true);
	obs_data_set_default_int(s, "log_level", LOG_DEBUG);
	obs_data_set_default_bool(s, "log_words", false);
	obs_data_set_default_string(s, "whisper_model_path", "Whisper Tiny English (74Mb)");
	obs_data_set_default_string(s, "whisper_language_select", "en");

	// Whisper parameters
	obs_data_set_default_int(s, "whisper_sampling_method", WHISPER_SAMPLING_BEAM_SEARCH);
	obs_data_set_default_string(s, "initial_prompt", "");
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
	obs_data_set_default_int(s, "max_tokens", 7);
	obs_data_set_default_bool(s, "speed_up", false);
	obs_data_set_default_bool(s, "suppress_blank", true);
	obs_data_set_default_bool(s, "suppress_non_speech_tokens", true);
	obs_data_set_default_double(s, "temperature", 0.1);
	obs_data_set_default_double(s, "max_initial_ts", 1.0);
	obs_data_set_default_double(s, "length_penalty", -1.0);
}

obs_properties_t *cleanstream_properties(void *data)
{
	UNUSED_PARAMETER(data);

	obs_properties_t *ppts = obs_properties_create();

	obs_properties_add_text(ppts, "detect_regex", MT_("detect_regex"), OBS_TEXT_DEFAULT);

	// Add a lift of available replace sounds
	obs_property_t *replace_sounds_list =
		obs_properties_add_list(ppts, "replace_sound", MT_("replace_sound"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(replace_sounds_list, "None", REPLACE_SOUNDS_NONE);
	obs_property_list_add_int(replace_sounds_list, "Silence", REPLACE_SOUNDS_SILENCE);
	// on windows and mac, add external file path for replace sound
#if defined(_WIN32) || defined(__APPLE__)
	struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);

	if (!gf->audioFileCache["beep.wav"].empty()) {
		obs_property_list_add_int(replace_sounds_list, "Beep", REPLACE_SOUNDS_BEEP);
	}
	if (!gf->audioFileCache["horn.wav"].empty()) {
		obs_property_list_add_int(replace_sounds_list, "Horn", REPLACE_SOUNDS_HORN);
	}
	obs_property_list_add_int(replace_sounds_list, "External", REPLACE_SOUNDS_EXTERNAL);

	// add external file path for replace sound
	obs_property_t *replace_sound_path = obs_properties_add_path(
		ppts, "replace_sound_path", MT_("replace_sound_path"), OBS_PATH_FILE,
		"WAV files (*.wav);;All files (*.*)", nullptr);

	// show/hide external file path based on the selected replace sound
	obs_property_set_modified_callback(replace_sounds_list, [](obs_properties_t *props,
								   obs_property_t *property,
								   obs_data_t *settings) {
		UNUSED_PARAMETER(property);
		const long long replace_sound = obs_data_get_int(settings, "replace_sound");
		obs_property_set_visible(obs_properties_get(props, "replace_sound_path"),
					 replace_sound == REPLACE_SOUNDS_EXTERNAL);
		return true;
	});

	obs_property_set_modified_callback2(
		replace_sound_path,
		[](void *data_, obs_properties_t *props, obs_property_t *property,
		   obs_data_t *settings) {
			UNUSED_PARAMETER(property);
			UNUSED_PARAMETER(props);
			struct cleanstream_data *gf_ =
				static_cast<struct cleanstream_data *>(data_);
			// load the sound file and cache it
			std::string replace_sound_path_ =
				obs_data_get_string(settings, "replace_sound_path");
			if (replace_sound_path_.empty()) {
				return true;
			}
			obs_log(LOG_INFO, "Loading audio file: %s", replace_sound_path_.c_str());
			AudioDataFloat audioFile =
				read_audio_file(replace_sound_path_.c_str(), gf_->sample_rate);
			obs_log(LOG_INFO, "Loaded %lu frames of audio data", audioFile.size());
			if (audioFile.empty()) {
				obs_log(LOG_ERROR, "Failed to load audio file: %s",
					replace_sound_path_.c_str());
				obs_data_set_string(settings, "replace_sound_path", "");
				return true;
			}
			gf_->audioFileCache[replace_sound_path_] = audioFile;
			gf_->replace_sound_external = replace_sound_path_;
			return true;
		},
		gf);
#endif

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

	// Add language selector
	obs_property_t *whisper_language_select_list =
		obs_properties_add_list(ppts, "whisper_language_select", "Language",
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	// get a sorted list of available languages
	std::vector<std::string> whisper_available_lang_keys;
	for (auto const &pair : whisper_available_lang) {
		whisper_available_lang_keys.push_back(pair.first);
	}
	std::sort(whisper_available_lang_keys.begin(), whisper_available_lang_keys.end());
	// iterate over all available languages in whisper_available_lang map<string, string>
	for (const std::string &key : whisper_available_lang_keys) {
		obs_property_list_add_string(whisper_language_select_list,
					     whisper_available_lang.at(key).c_str(), key.c_str());
	}

	// Add advanced settings checkbox
	obs_property_t *advanced_settings_prop =
		obs_properties_add_bool(ppts, "advanced_settings", MT_("advanced_settings"));
	obs_property_set_modified_callback(advanced_settings_prop, [](obs_properties_t *props,
								      obs_property_t *property,
								      obs_data_t *settings) {
		UNUSED_PARAMETER(property);
		// If advanced settings is enabled, show the advanced settings group
		const bool show_hide = obs_data_get_bool(settings, "advanced_settings");
		for (const std::string &prop_name :
		     {"whisper_params_group", "log_words", "vad_enabled", "log_level"}) {
			obs_property_set_visible(obs_properties_get(props, prop_name.c_str()),
						 show_hide);
		}
		return true;
	});

	obs_properties_add_bool(ppts, "vad_enabled", MT_("vad_enabled"));
	obs_property_t *list = obs_properties_add_list(ppts, "log_level", MT_("log_level"),
						       OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(list, "DEBUG", LOG_DEBUG);
	obs_property_list_add_int(list, "INFO", LOG_INFO);
	obs_property_list_add_int(list, "WARNING", LOG_WARNING);
	obs_properties_add_bool(ppts, "log_words", MT_("log_words"));

	obs_properties_t *whisper_params_group = obs_properties_create();
	obs_properties_add_group(ppts, "whisper_params_group", MT_("Whisper_Parameters"),
				 OBS_GROUP_NORMAL, whisper_params_group);

	obs_property_t *whisper_sampling_method_list = obs_properties_add_list(
		whisper_params_group, "whisper_sampling_method", MT_("whisper_sampling_method"),
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(whisper_sampling_method_list, "Beam search",
				  WHISPER_SAMPLING_BEAM_SEARCH);
	obs_property_list_add_int(whisper_sampling_method_list, "Greedy", WHISPER_SAMPLING_GREEDY);

	// int n_threads;
	obs_properties_add_int_slider(whisper_params_group, "n_threads", MT_("n_threads"), 1, 8, 1);
	// int n_max_text_ctx;     // max tokens to use from past text as prompt for the decoder
	obs_properties_add_int_slider(whisper_params_group, "n_max_text_ctx", MT_("n_max_text_ctx"),
				      0, 16384, 100);
	// int offset_ms;          // start offset in ms
	// int duration_ms;        // audio duration to process in ms
	// bool translate;
	// bool no_context;        // do not use past transcription (if any) as initial prompt for the decoder
	obs_properties_add_bool(whisper_params_group, "no_context", MT_("no_context"));
	// bool single_segment;    // force single segment output (useful for streaming)
	obs_properties_add_bool(whisper_params_group, "single_segment", MT_("single_segment"));
	// bool print_special;     // print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)
	obs_properties_add_bool(whisper_params_group, "print_special", MT_("print_special"));
	// bool print_progress;    // print progress information
	obs_properties_add_bool(whisper_params_group, "print_progress", MT_("print_progress"));
	// bool print_realtime;    // print results from within whisper.cpp (avoid it, use callback instead)
	obs_properties_add_bool(whisper_params_group, "print_realtime", MT_("print_realtime"));
	// bool print_timestamps;  // print timestamps for each text segment when printing realtime
	obs_properties_add_bool(whisper_params_group, "print_timestamps", MT_("print_timestamps"));
	// bool  token_timestamps; // enable token-level timestamps
	obs_properties_add_bool(whisper_params_group, "token_timestamps", MT_("token_timestamps"));
	// float thold_pt;         // timestamp token probability threshold (~0.01)
	obs_properties_add_float_slider(whisper_params_group, "thold_pt", MT_("thold_pt"), 0.0f,
					1.0f, 0.05f);
	// float thold_ptsum;      // timestamp token sum probability threshold (~0.01)
	obs_properties_add_float_slider(whisper_params_group, "thold_ptsum", MT_("thold_ptsum"),
					0.0f, 1.0f, 0.05f);
	// int   max_len;          // max segment length in characters
	obs_properties_add_int_slider(whisper_params_group, "max_len", MT_("max_len"), 0, 100, 1);
	// bool  split_on_word;    // split on word rather than on token (when used with max_len)
	obs_properties_add_bool(whisper_params_group, "split_on_word", MT_("split_on_word"));
	// int   max_tokens;       // max tokens per segment (0 = no limit)
	obs_properties_add_int_slider(whisper_params_group, "max_tokens", MT_("max_tokens"), 0, 100,
				      1);
	// bool speed_up;          // speed-up the audio by 2x using Phase Vocoder
	obs_properties_add_bool(whisper_params_group, "speed_up", MT_("speed_up"));
	// const char * initial_prompt;
	obs_properties_add_text(whisper_params_group, "initial_prompt", MT_("initial_prompt"),
				OBS_TEXT_DEFAULT);
	// bool suppress_blank
	obs_properties_add_bool(whisper_params_group, "suppress_blank", MT_("suppress_blank"));
	// bool suppress_non_speech_tokens
	obs_properties_add_bool(whisper_params_group, "suppress_non_speech_tokens",
				MT_("suppress_non_speech_tokens"));
	// float temperature
	obs_properties_add_float_slider(whisper_params_group, "temperature", MT_("temperature"),
					0.0f, 1.0f, 0.05f);
	// float max_initial_ts
	obs_properties_add_float_slider(whisper_params_group, "max_initial_ts",
					MT_("max_initial_ts"), 0.0f, 1.0f, 0.05f);
	// float length_penalty
	obs_properties_add_float_slider(whisper_params_group, "length_penalty",
					MT_("length_penalty"), -1.0f, 1.0f, 0.1f);

	return ppts;
}
