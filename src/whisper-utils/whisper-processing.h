#ifndef WHISPER_PROCESSING_H
#define WHISPER_PROCESSING_H

// buffer size in msec
#define DEFAULT_BUFFER_SIZE_MSEC 3000
// overlap in msec
#define DEFAULT_OVERLAP_SIZE_MSEC 100

enum DetectionResult {
	DETECTION_RESULT_UNKNOWN = 0,
	DETECTION_RESULT_SILENCE = 1,
	DETECTION_RESULT_SPEECH = 2,
	DETECTION_RESULT_FILLER = 3,
	DETECTION_RESULT_BEEP = 4,
};

void whisper_loop(void *data);
struct whisper_context *init_whisper_context(const std::string &model_path_in,
					     struct cleanstream_data *gf);

#endif // WHISPER_PROCESSING_H
