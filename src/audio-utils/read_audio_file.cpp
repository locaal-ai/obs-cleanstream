extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/mem.h>
#include <libswresample/swresample.h>
}

#include <iostream>
#include <vector>

std::vector<std::vector<float>> read_audio_file(const char *filename, int targetSampleRate = 48000)
{

	AVFormatContext *formatContext = nullptr;
	if (avformat_open_input(&formatContext, filename, nullptr, nullptr) != 0) {
		std::cerr << "Error opening file\n";
		return {};
	}

	if (avformat_find_stream_info(formatContext, nullptr) < 0) {
		std::cerr << "Error finding stream information\n";
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
		std::cerr << "No audio stream found\n";
		return {};
	}

	AVCodecParameters *codecParams = formatContext->streams[audioStreamIndex]->codecpar;
	const AVCodec *codec = avcodec_find_decoder(codecParams->codec_id);
	if (!codec) {
		std::cerr << "Decoder not found\n";
		return {};
	}

	AVCodecContext *codecContext = avcodec_alloc_context3(codec);
	if (!codecContext) {
		std::cerr << "Failed to allocate codec context\n";
		return {};
	}

	if (avcodec_parameters_to_context(codecContext, codecParams) < 0) {
		std::cerr << "Failed to copy codec parameters to codec context\n";
		return {};
	}

	if (avcodec_open2(codecContext, codec, nullptr) < 0) {
		std::cerr << "Failed to open codec\n";
		return {};
	}

	SwrContext *swrContext = swr_alloc_set_opts(
		nullptr, av_get_default_channel_layout(codecParams->channels),
		AV_SAMPLE_FMT_FLT, // Output sample format (float)
		targetSampleRate, av_get_default_channel_layout(codecParams->channels),
		static_cast<AVSampleFormat>(codecParams->format), codecParams->sample_rate, 0,
		nullptr);

	if (!swrContext || swr_init(swrContext) < 0) {
		std::cerr << "Failed to initialize the resampling context\n";
		return {};
	}

	AVFrame *frame = av_frame_alloc();
	AVFrame *resampledFrame = av_frame_alloc();
	AVPacket packet;

	std::vector<std::vector<float>> audioFrames;

	while (av_read_frame(formatContext, &packet) >= 0) {
		if (packet.stream_index == audioStreamIndex) {
			if (avcodec_send_packet(codecContext, &packet) == 0) {
				while (avcodec_receive_frame(codecContext, frame) == 0) {
					swr_convert_frame(swrContext, resampledFrame, frame);

					int num_samples = resampledFrame->nb_samples;
					int num_channels = codecParams->channels;
					std::vector<float> buffer(num_samples * num_channels);
					memcpy(buffer.data(), resampledFrame->data[0],
					       buffer.size() * sizeof(float));
					audioFrames.push_back(buffer);
				}
			}
		}
		av_packet_unref(&packet);
	}

	av_frame_free(&frame);
	av_frame_free(&resampledFrame);
	avcodec_free_context(&codecContext);
	swr_free(&swrContext);
	avformat_close_input(&formatContext);
	avformat_free_context(formatContext);

	// Audio frames are now stored in audioFrames with float samples

	return audioFrames;
}
