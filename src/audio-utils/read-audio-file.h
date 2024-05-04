#ifndef READ_AUDIO_FILE_H
#define READ_AUDIO_FILE_H

#include <vector>
#include <media-io/audio-io.h>

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

typedef std::vector<float> AudioDataFloat;

AudioDataFloat read_audio_file(const char *filename, int targetSampleRate = 48000);

#endif // READ_AUDIO_FILE_H
