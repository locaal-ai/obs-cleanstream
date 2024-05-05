#include <whisper.h>

#include <obs-module.h>

#include "plugin-support.h"
#include "cleanstream-filter-data.h"
#include "whisper-processing.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <filesystem>
#include <regex>

#ifdef _WIN32
#include <fstream>
#define NOMINMAX
#include <Windows.h>
#endif
#include "model-utils/model-downloader.h"

#define VAD_THOLD 0.0001f
#define FREQ_THOLD 100.0f

void high_pass_filter(float *pcmf32, size_t pcm32f_size, float cutoff, uint32_t sample_rate)
{
	const float rc = 1.0f / (2.0f * (float)M_PI * cutoff);
	const float dt = 1.0f / (float)sample_rate;
	const float alpha = dt / (rc + dt);

	float y = pcmf32[0];

	for (size_t i = 1; i < pcm32f_size; i++) {
		y = alpha * (y + pcmf32[i] - pcmf32[i - 1]);
		pcmf32[i] = y;
	}
}

// VAD (voice activity detection), return true if speech detected
bool vad_simple(float *pcmf32, size_t pcm32f_size, uint32_t sample_rate, float vad_thold,
		float freq_thold, bool verbose)
{
	const uint64_t n_samples = pcm32f_size;

	if (freq_thold > 0.0f) {
		high_pass_filter(pcmf32, pcm32f_size, freq_thold, sample_rate);
	}

	float energy_all = 0.0f;

	for (uint64_t i = 0; i < n_samples; i++) {
		energy_all += fabsf(pcmf32[i]);
	}

	energy_all /= (float)n_samples;

	if (verbose) {
		obs_log(LOG_INFO, "%s: energy_all: %f, vad_thold: %f, freq_thold: %f", __func__,
			energy_all, vad_thold, freq_thold);
	}

	if (energy_all < vad_thold) {
		return false;
	}

	return true;
}

float avg_energy_in_window(const float *pcmf32, size_t window_i, uint64_t n_samples_window)
{
	float energy_in_window = 0.0f;
	for (uint64_t j = 0; j < n_samples_window; j++) {
		energy_in_window += fabsf(pcmf32[window_i + j]);
	}
	energy_in_window /= (float)n_samples_window;

	return energy_in_window;
}

float max_energy_in_window(const float *pcmf32, size_t window_i, uint64_t n_samples_window)
{
	float energy_in_window = 0.0f;
	for (uint64_t j = 0; j < n_samples_window; j++) {
		energy_in_window = std::max(energy_in_window, fabsf(pcmf32[window_i + j]));
	}

	return energy_in_window;
}

// Find a word boundary
size_t word_boundary_simple(const float *pcmf32, size_t pcm32f_size, uint32_t sample_rate,
			    float thold, bool verbose)
{
	// scan the buffer with a window of 50ms
	const uint64_t n_samples_window = (sample_rate * 50) / 1000;

	float first_window_energy = avg_energy_in_window(pcmf32, 0, n_samples_window);
	float last_window_energy =
		avg_energy_in_window(pcmf32, pcm32f_size - n_samples_window, n_samples_window);
	float max_energy_in_middle =
		max_energy_in_window(pcmf32, n_samples_window, pcm32f_size - n_samples_window);

	if (verbose) {
		obs_log(LOG_INFO,
			"%s: first_window_energy: %f, last_window_energy: %f, max_energy_in_middle: %f",
			__func__, first_window_energy, last_window_energy, max_energy_in_middle);
		// print avg energy in all windows in sample
		for (uint64_t i = 0; i < pcm32f_size - n_samples_window; i += n_samples_window) {
			obs_log(LOG_INFO, "%s: avg energy_in_window %llu: %f", __func__, i,
				avg_energy_in_window(pcmf32, i, n_samples_window));
		}
	}

	const float max_energy_thold = max_energy_in_middle * thold;
	if (first_window_energy < max_energy_thold && last_window_energy < max_energy_thold) {
		if (verbose) {
			obs_log(LOG_INFO, "%s: word boundary found between %llu and %llu", __func__,
				n_samples_window, pcm32f_size - n_samples_window);
		}
		return n_samples_window;
	}

	return 0;
}

struct whisper_context *init_whisper_context(const std::string &model_path_in,
					     struct cleanstream_data *gf)
{
	std::string model_path = model_path_in;

	obs_log(LOG_INFO, "Loading whisper model from %s", model_path.c_str());

	if (std::filesystem::is_directory(model_path)) {
		obs_log(LOG_INFO,
			"Model path is a directory, not a file, looking for .bin file in folder");
		// look for .bin file
		const std::string model_bin_file = find_bin_file_in_folder(model_path);
		if (model_bin_file.empty()) {
			obs_log(LOG_ERROR, "Model bin file not found in folder: %s",
				model_path.c_str());
			return nullptr;
		}
		model_path = model_bin_file;
	}

	whisper_log_set(
		[](enum ggml_log_level level, const char *text, void *user_data) {
			UNUSED_PARAMETER(level);
			struct cleanstream_data *ctx =
				static_cast<struct cleanstream_data *>(user_data);
			// remove trailing newline
			char *text_copy = bstrdup(text);
			text_copy[strcspn(text_copy, "\n")] = 0;
			obs_log(ctx->log_level, "Whisper: %s", text_copy);
			bfree(text_copy);
		},
		gf);

	struct whisper_context_params cparams = whisper_context_default_params();
#ifdef LOCALVOCAL_WITH_CUDA
	cparams.use_gpu = true;
	cparams.gpu_device = 0;
	obs_log(LOG_INFO, "Using CUDA GPU for inference, device %d", cparams.gpu_device);
#elif defined(LOCALVOCAL_WITH_CLBLAST)
	cparams.use_gpu = true;
	cparams.gpu_device = 0;
	obs_log(LOG_INFO, "Using OpenCL for inference");
#else
	cparams.use_gpu = false;
	obs_log(LOG_INFO, "Using CPU for inference");
#endif

	struct whisper_context *ctx = nullptr;
	try {
#ifdef _WIN32
		// convert model path UTF8 to wstring (wchar_t) for whisper
		int count = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(),
						(int)model_path.length(), NULL, 0);
		std::wstring model_path_ws(count, 0);
		MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), (int)model_path.length(),
				    &model_path_ws[0], count);

		// Read model into buffer
		std::ifstream modelFile(model_path_ws, std::ios::binary);
		if (!modelFile.is_open()) {
			obs_log(LOG_ERROR, "Failed to open whisper model file %s",
				model_path.c_str());
			return nullptr;
		}
		modelFile.seekg(0, std::ios::end);
		const size_t modelFileSize = modelFile.tellg();
		modelFile.seekg(0, std::ios::beg);
		std::vector<char> modelBuffer(modelFileSize);
		modelFile.read(modelBuffer.data(), modelFileSize);
		modelFile.close();

		// Initialize whisper
		ctx = whisper_init_from_buffer_with_params(modelBuffer.data(), modelFileSize,
							   cparams);
#else
		ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
#endif
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "Exception while loading whisper model: %s", e.what());
		return nullptr;
	}
	if (ctx == nullptr) {
		obs_log(LOG_ERROR, "Failed to load whisper model");
		return nullptr;
	}

	obs_log(LOG_INFO, "Whisper model loaded: %s", whisper_print_system_info());
	return ctx;
}

std::string to_timestamp(int64_t t)
{
	int64_t sec = t / 100;
	int64_t msec = t - sec * 100;
	int64_t min = sec / 60;
	sec = sec - min * 60;

	char buf[32];
	snprintf(buf, sizeof(buf), "%02d:%02d.%03d", (int)min, (int)sec, (int)msec);

	return std::string(buf);
}

int run_whisper_inference(struct cleanstream_data *gf, const float *pcm32f_data, size_t pcm32f_size)
{
	std::lock_guard<std::mutex> lock(gf->whisper_ctx_mutex);

	if (gf->whisper_context == nullptr) {
		obs_log(LOG_WARNING, "whisper context is null");
		return DETECTION_RESULT_UNKNOWN;
	}

	obs_log(gf->log_level, "%s: processing %d samples, %.3f sec, %d threads", __func__,
		int(pcm32f_size), float(pcm32f_size) / WHISPER_SAMPLE_RATE,
		gf->whisper_params.n_threads);

	// run the inference
	int whisper_full_result = -1;
	try {
		whisper_full_result = whisper_full(gf->whisper_context, gf->whisper_params,
						   pcm32f_data, (int)pcm32f_size);
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "Whisper exception: %s. Filter restart is required", e.what());
		whisper_free(gf->whisper_context);
		gf->whisper_context = nullptr;
		return DETECTION_RESULT_UNKNOWN;
	}

	if (whisper_full_result != 0) {
		obs_log(LOG_WARNING, "failed to process audio, error %d", whisper_full_result);
		return DETECTION_RESULT_UNKNOWN;
	} else {
		const int n_segment = 0;
		const char *text = whisper_full_get_segment_text(gf->whisper_context, n_segment);
		const int64_t t0 = whisper_full_get_segment_t0(gf->whisper_context, n_segment);
		const int64_t t1 = whisper_full_get_segment_t1(gf->whisper_context, n_segment);

		float sentence_p = 0.0f;
		const int n_tokens = whisper_full_n_tokens(gf->whisper_context, n_segment);
		for (int j = 0; j < n_tokens; ++j) {
			sentence_p += whisper_full_get_token_p(gf->whisper_context, n_segment, j);
		}
		sentence_p /= (float)n_tokens;

		// convert text to lowercase
		std::string text_lower(text);
		std::transform(text_lower.begin(), text_lower.end(), text_lower.begin(), ::tolower);
		// trim whitespace (use lambda)
		text_lower.erase(std::find_if(text_lower.rbegin(), text_lower.rend(),
					      [](unsigned char ch) { return !std::isspace(ch); })
					 .base(),
				 text_lower.end());

		if (gf->log_words) {
			obs_log(LOG_INFO, "[%s --> %s] (%.3f) %s", to_timestamp(t0).c_str(),
				to_timestamp(t1).c_str(), sentence_p, text_lower.c_str());
		}

		if (text_lower.empty()) {
			return DETECTION_RESULT_SILENCE;
		}

		// use a regular expression to detect filler words with a word boundary
		try {
			if (gf->detect_regex != nullptr && strlen(gf->detect_regex) > 0) {
				std::regex filler_regex(gf->detect_regex);
				if (std::regex_search(text_lower, filler_regex,
						      std::regex_constants::match_any)) {
					return DETECTION_RESULT_BEEP;
				}
			}
		} catch (const std::regex_error &e) {
			obs_log(LOG_ERROR, "Regex error: %s", e.what());
		}
	}

	return DETECTION_RESULT_SPEECH;
}

long long process_audio_from_buffer(struct cleanstream_data *gf)
{
	uint64_t start_timestamp = 0;

	{
		// scoped lock the buffer mutex
		std::lock_guard<std::mutex> lock(gf->whisper_buf_mutex);

		// copy gf->frames from the end of the input buffer to the copy_buffers
		for (size_t c = 0; c < gf->channels; c++) {
			circlebuf_peek_front(&gf->input_buffers[c], gf->copy_buffers[c],
					     gf->frames * sizeof(float));
		}

		// peek at the info_buffer to get the timestamp of the first info
		struct cleanstream_audio_info info_from_buf = {0};
		circlebuf_peek_front(&gf->info_buffer, &info_from_buf,
				     sizeof(struct cleanstream_audio_info));
		start_timestamp = info_from_buf.timestamp;
	}

	obs_log(gf->log_level, "processing %lu frames (%d ms), start timestamp %llu ", gf->frames,
		(int)(gf->frames * 1000 / gf->sample_rate), start_timestamp);

	// time the audio processing
	auto start = std::chrono::high_resolution_clock::now();

	// resample to 16kHz
	float *whisper_buffer_16khz[MAX_PREPROC_CHANNELS];
	uint32_t whisper_frames;
	uint64_t ts_offset;
	audio_resampler_resample(gf->resampler, (uint8_t **)whisper_buffer_16khz, &whisper_frames,
				 &ts_offset, (const uint8_t **)gf->copy_buffers,
				 (uint32_t)gf->frames);

	obs_log(gf->log_level, "%d channels, %d whisper frames, %f ms", (int)gf->channels,
		(int)whisper_frames, (float)whisper_frames / WHISPER_SAMPLE_RATE * 1000.0f);

	bool skipped_inference = false;

	if (gf->vad_enabled && gf->vad != nullptr) {
		std::vector<float> vad_input(whisper_buffer_16khz[0],
					     whisper_buffer_16khz[0] + whisper_frames);
		gf->vad->process(vad_input);

		std::vector<timestamp_t> stamps = gf->vad->get_speech_timestamps();
		if (stamps.size() == 0) {
			obs_log(gf->log_level, "VAD detected no speech in %d frames",
				whisper_buffer_16khz);
			skipped_inference = true;
		}
	}

	if (!skipped_inference) {
		// run inference
		const int inference_result =
			run_whisper_inference(gf, whisper_buffer_16khz[0], whisper_frames);
		{
			std::lock_guard<std::mutex> lock(gf->whisper_outbuf_mutex);
			gf->current_result = inference_result;
		}
	} else {
		if (gf->log_words) {
			obs_log(LOG_INFO, "skipping inference");
		}
	}

	// end of timer
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	const uint32_t audio_processed_ms =
		(uint32_t)gf->frames * 1000u / gf->sample_rate; // number of frames in this packet
	obs_log(gf->log_level, "audio processing of %u ms new data took %d ms", audio_processed_ms,
		(int)duration);

	return duration;
}

void whisper_loop(void *data)
{
	struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);
	const size_t segment_size = gf->frames * sizeof(float);

	obs_log(LOG_INFO, "starting whisper thread");

	// Thread main loop
	while (true) {
		{
			std::lock_guard<std::mutex> lock(gf->whisper_ctx_mutex);
			if (gf->whisper_context == nullptr) {
				obs_log(LOG_WARNING, "Whisper context is null, exiting thread");
				break;
			}
		}

		// Check if we have enough data to process
		size_t input_buf_size = 0;
		{
			// std::lock_guard<std::mutex> lock(gf->whisper_buf_mutex);
			input_buf_size = gf->input_buffers[0].size;
		}

		obs_log(gf->log_level,
			"found %lu bytes, %lu frames in input buffer, need >= %lu, processing? %d",
			input_buf_size, (size_t)(input_buf_size / sizeof(float)), segment_size,
			input_buf_size >= segment_size);

		long long duration = 0;
		if (input_buf_size >= segment_size) {

			// Process the audio. This will also remove the processed data from the input buffer.
			// Mutex is locked inside process_audio_from_buffer.
			duration = process_audio_from_buffer(gf);
		}
		// sleep for up to 300ms depending on the processing time
		if (duration < 300) {
			std::this_thread::sleep_for(std::chrono::milliseconds(300 - duration));
		}
	}

	obs_log(LOG_INFO, "exiting whisper thread");
}
