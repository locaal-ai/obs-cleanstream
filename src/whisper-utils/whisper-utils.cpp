#include "whisper-utils.h"
#include "plugin-support.h"
#include "model-utils/model-downloader.h"
#include "whisper-processing.h"

#include <obs-module.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

void update_whisper_model(struct cleanstream_data *gf, obs_data_t *s)
{
	// update the whisper model path
	std::string new_model_path = obs_data_get_string(s, "whisper_model_path");
	const bool is_external_model = new_model_path.find("!!!external!!!") != std::string::npos;

	if (gf->whisper_model_path.empty() || gf->whisper_model_path != new_model_path ||
	    is_external_model) {

		if (gf->whisper_model_path != new_model_path) {
			// model path changed
			obs_log(gf->log_level, "model path changed from %s to %s",
				gf->whisper_model_path.c_str(), new_model_path.c_str());
		}

		// check if the new model is external file
		if (!is_external_model) {
			// new model is not external file
			shutdown_whisper_thread(gf);

			if (models_info.count(new_model_path) == 0) {
				obs_log(LOG_WARNING, "Model '%s' does not exist",
					new_model_path.c_str());
				return;
			}

			const ModelInfo &model_info = models_info[new_model_path];

			// check if the model exists, if not, download it
			std::string model_file_found = find_model_bin_file(model_info);
			if (model_file_found == "") {
				obs_log(LOG_WARNING, "Whisper model does not exist");
				download_model_with_ui_dialog(
					model_info, [gf, new_model_path](int download_status,
									 const std::string &path) {
						if (download_status == 0) {
							obs_log(LOG_INFO,
								"Model download complete");
							gf->whisper_model_path = new_model_path;
							start_whisper_thread_with_path(gf, path);
						} else {
							obs_log(LOG_ERROR, "Model download failed");
						}
					});
			} else {
				// Model exists, just load it
				gf->whisper_model_path = new_model_path;
				start_whisper_thread_with_path(gf, model_file_found);
			}
		} else {
			// new model is external file, get file location from file property
			std::string external_model_file_path =
				obs_data_get_string(s, "whisper_model_path_external");
			if (external_model_file_path.empty()) {
				obs_log(LOG_WARNING, "External model file path is empty");
			} else {
				// check if the external model file is not currently loaded
				if (gf->whisper_model_file_currently_loaded ==
				    external_model_file_path) {
					obs_log(LOG_INFO, "External model file is already loaded");
					return;
				} else {
					shutdown_whisper_thread(gf);
					gf->whisper_model_path = new_model_path;
					start_whisper_thread_with_path(gf,
								       external_model_file_path);
				}
			}
		}
	} else {
		// model path did not change
		obs_log(gf->log_level, "Model path did not change: %s == %s",
			gf->whisper_model_path.c_str(), new_model_path.c_str());
	}
}

void shutdown_whisper_thread(struct cleanstream_data *gf)
{
	obs_log(gf->log_level, "shutdown_whisper_thread");
	if (gf->whisper_context != nullptr) {
		// acquire the mutex before freeing the context
		std::lock_guard<std::mutex> lock(gf->whisper_ctx_mutex);
		whisper_free(gf->whisper_context);
		gf->whisper_context = nullptr;
	}
	if (gf->whisper_thread.joinable()) {
		gf->whisper_thread.join();
	}
	if (!gf->whisper_model_path.empty()) {
		gf->whisper_model_path = "";
	}
}

void start_whisper_thread_with_path(struct cleanstream_data *gf, const std::string &path)
{
	obs_log(gf->log_level, "start_whisper_thread_with_path: %s", path.c_str());
	std::lock_guard<std::mutex> lock(gf->whisper_ctx_mutex);
	if (gf->whisper_context != nullptr) {
		obs_log(LOG_ERROR, "cannot init whisper: whisper_context is not null");
		return;
	}

	// initialize Silero VAD
	char *data_folder_models = obs_module_file("models");
	if (data_folder_models == nullptr) {
		obs_log(LOG_ERROR, "Failed to find models folder");
		return;
	}
	const std::filesystem::path module_data_models_folder =
		std::filesystem::absolute(data_folder_models);
	bfree(data_folder_models);
#ifdef _WIN32
	std::wstring silero_vad_model_path =
		module_data_models_folder.wstring() + L"\\silero-vad\\silero_vad.onnx";
	obs_log(gf->log_level, "silero vad model path: %ls", silero_vad_model_path.c_str());
#else
	std::string silero_vad_model_path =
		module_data_models_folder.string() + "/silero-vad/silero_vad.onnx";
	obs_log(gf->log_level, "silero vad model path: %s", silero_vad_model_path.c_str());
#endif
	// roughly following https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py
	// for silero vad parameters
	gf->vad.reset(new VadIterator(silero_vad_model_path, WHISPER_SAMPLE_RATE, 64, 0.5f, 1000,
				      200, 250));

	gf->whisper_context = init_whisper_context(path, gf);
	if (gf->whisper_context == nullptr) {
		obs_log(LOG_ERROR, "Failed to initialize whisper context");
		return;
	}
	gf->whisper_model_file_currently_loaded = path;
	std::thread new_whisper_thread(whisper_loop, gf);
	gf->whisper_thread.swap(new_whisper_thread);
}

#define is_lead_byte(c) (((c) & 0xe0) == 0xc0 || ((c) & 0xf0) == 0xe0 || ((c) & 0xf8) == 0xf0)
#define is_trail_byte(c) (((c) & 0xc0) == 0x80)

inline int lead_byte_length(const uint8_t c)
{
	if ((c & 0xe0) == 0xc0) {
		return 2;
	} else if ((c & 0xf0) == 0xe0) {
		return 3;
	} else if ((c & 0xf8) == 0xf0) {
		return 4;
	} else {
		return 1;
	}
}

inline bool is_valid_lead_byte(const uint8_t *c)
{
	const int length = lead_byte_length(c[0]);
	if (length == 1) {
		return true;
	}
	if (length == 2 && is_trail_byte(c[1])) {
		return true;
	}
	if (length == 3 && is_trail_byte(c[1]) && is_trail_byte(c[2])) {
		return true;
	}
	if (length == 4 && is_trail_byte(c[1]) && is_trail_byte(c[2]) && is_trail_byte(c[3])) {
		return true;
	}
	return false;
}

/*
* Fix UTF8 encoding issues on Windows.
*/
std::string fix_utf8(const std::string &str)
{
#ifdef _WIN32
	// Some UTF8 charsets on Windows output have a bug, instead of 0xd? it outputs
	// 0xf?, and 0xc? becomes 0xe?, so we need to fix it.
	std::stringstream ss;
	uint8_t *c_str = (uint8_t *)str.c_str();
	for (size_t i = 0; i < str.size(); ++i) {
		if (is_lead_byte(c_str[i])) {
			// this is a unicode leading byte
			// if the next char is 0xff - it's a bug char, replace it with 0x9f
			if (c_str[i + 1] == 0xff) {
				c_str[i + 1] = 0x9f;
			}
			if (!is_valid_lead_byte(c_str + i)) {
				// This is a bug lead byte, because it's length 3 and the i+2 byte is also
				// a lead byte
				c_str[i] = c_str[i] - 0x20;
			}
		} else {
			if (c_str[i] >= 0xf8) {
				// this may be a malformed lead byte.
				// lets see if it becomes a valid lead byte if we "fix" it
				uint8_t buf_[4];
				buf_[0] = c_str[i] - 0x20;
				buf_[1] = c_str[i + 1];
				buf_[2] = c_str[i + 2];
				buf_[3] = c_str[i + 3];
				if (is_valid_lead_byte(buf_)) {
					// this is a malformed lead byte, fix it
					c_str[i] = c_str[i] - 0x20;
				}
			}
		}
	}

	return std::string((char *)c_str);
#else
	return str;
#endif
}

/*
* Remove leading and trailing non-alphabetic characters from a string.
* This function is used to remove leading and trailing spaces, newlines, tabs or punctuation.
* @param str: the string to remove leading and trailing non-alphabetic characters from.
* @return: the string with leading and trailing non-alphabetic characters removed.
*/
std::string remove_leading_trailing_nonalpha(const std::string &str)
{
	std::string str_copy = str;
	// remove trailing spaces, newlines, tabs or punctuation
	auto last_non_space =
		std::find_if(str_copy.rbegin(), str_copy.rend(), [](unsigned char ch) {
			return !std::isspace(ch) || !std::ispunct(ch);
		}).base();
	str_copy.erase(last_non_space, str_copy.end());
	// remove leading spaces, newlines, tabs or punctuation
	auto first_non_space = std::find_if(str_copy.begin(), str_copy.end(),
					    [](unsigned char ch) {
						    return !std::isspace(ch) || !std::ispunct(ch);
					    }) +
			       1;
	str_copy.erase(str_copy.begin(), first_non_space);
	return str_copy;
}
