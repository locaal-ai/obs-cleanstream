extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

#include <iostream>
#include <vector>

#include "read-audio-file.h"
#include "obs.h"
#include "plugin-support.h"

AudioDataFloat read_audio_file(const char *filename, int targetSampleRate)
{
	AVFormatContext *formatContext = nullptr;
	if (avformat_open_input(&formatContext, filename, nullptr, nullptr) != 0) {
		obs_log(LOG_ERROR, "Error opening file");
		return {};
	}

	if (avformat_find_stream_info(formatContext, nullptr) < 0) {
		obs_log(LOG_ERROR, "Error finding stream information");
		return {};
	}

	int audioStreamIndex = -1;
	for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
		if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
			audioStreamIndex = i;
			break;
		}
	}

	if (audioStreamIndex == -1) {
		obs_log(LOG_ERROR, "No audio stream found");
		return {};
	}

	AVCodecParameters *codecParams = formatContext->streams[audioStreamIndex]->codecpar;
	const AVCodec *codec = avcodec_find_decoder(codecParams->codec_id);
	if (!codec) {
		obs_log(LOG_ERROR, "Decoder not found");
		return {};
	}

	AVCodecContext *codecContext = avcodec_alloc_context3(codec);
	if (!codecContext) {
		obs_log(LOG_ERROR, "Failed to allocate codec context");
		return {};
	}

	if (avcodec_parameters_to_context(codecContext, codecParams) < 0) {
		obs_log(LOG_ERROR, "Failed to copy codec parameters to codec context");
		return {};
	}

	if (avcodec_open2(codecContext, codec, nullptr) < 0) {
		obs_log(LOG_ERROR, "Failed to open codec");
		return {};
	}

	AVFrame *frame = av_frame_alloc();
	AVPacket packet;

	// set up swresample
	AVChannelLayout ch_layout;
	av_channel_layout_from_string(&ch_layout, "mono");
	SwrContext *swr = nullptr;
	int ret;
	ret = swr_alloc_set_opts2(&swr, &ch_layout, AV_SAMPLE_FMT_FLT, targetSampleRate,
				  &(codecContext->ch_layout), codecContext->sample_fmt,
				  codecContext->sample_rate, 0, nullptr);
	if (ret < 0) {
		char errbuf[AV_ERROR_MAX_STRING_SIZE];
		av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
		obs_log(LOG_ERROR, "Failed to set up swr context: %s", errbuf);
		return {};
	}
	// init swr
	ret = swr_init(swr);
	if (ret < 0) {
		char errbuf[AV_ERROR_MAX_STRING_SIZE];
		av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
		obs_log(LOG_ERROR, "Failed to initialize swr context: %s", errbuf);
		return {};
	}

	AudioDataFloat audioFrames;

	float *convertBuffer[1];
	convertBuffer[0] = (float *)av_malloc(4096 * sizeof(float));
	while (av_read_frame(formatContext, &packet) >= 0) {
		if (packet.stream_index == audioStreamIndex) {
			if (avcodec_send_packet(codecContext, &packet) == 0) {
				while (avcodec_receive_frame(codecContext, frame) == 0) {
					int ret = swr_convert(swr, (uint8_t **)convertBuffer, 4096,
							      (const uint8_t **)frame->data,
							      frame->nb_samples);
					if (ret < 0) {
						char errbuf[AV_ERROR_MAX_STRING_SIZE];
						av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
						obs_log(LOG_ERROR,
							"Failed to convert audio frame: %s",
							errbuf);
						return {};
					}
					audioFrames.insert(audioFrames.end(), convertBuffer[0],
							   convertBuffer[0] + ret);
				}
			}
		}
		av_packet_unref(&packet);
	}
	av_free(convertBuffer[0]);

	obs_log(LOG_INFO,
		"Converted %lu frames of audio data (orig: %d, %s sample format, %d channels, %s)",
		audioFrames.size(), codecContext->sample_rate,
		av_get_sample_fmt_name(codecContext->sample_fmt),
		codecContext->ch_layout.nb_channels,
		av_sample_fmt_is_planar(codecContext->sample_fmt) ? "planar" : "packed");

	swr_free(&swr);
	av_frame_free(&frame);
	avcodec_free_context(&codecContext);
	avformat_close_input(&formatContext);

	return audioFrames;
}
