#include <obs-module.h>
#include <media-io/audio-math.h>
#include <media-io/audio-resampler.h>
#include <util/circlebuf.h>
#include <math.h>

#include <string>
#include <thread>
#include <mutex>

#include <whisper.h>

#define do_log(level, format, ...) \
  blog(level, "[cleanstream filter: '%s'] " format, obs_source_get_name(gf->context), ##__VA_ARGS__)

#define warn(format, ...) do_log(LOG_WARNING, format, ##__VA_ARGS__)
#define info(format, ...) do_log(LOG_INFO, format, ##__VA_ARGS__)
#define debug(format, ...) do_log(LOG_DEBUG, format, ##__VA_ARGS__)

#define MAX_PREPROC_CHANNELS 8

// buffer size in msec
#define BUFFER_SIZE_MSEC 1010
// at 16Khz, 1001 msec is 16016 frames
#define WHISPER_FRAME_SIZE 16160
// 100ms overlap
#define OVERLAP_SIZE_MSEC 100

#define VAD_THOLD 0.0001f
#define FREQ_THOLD 100.0f

#define S_cleanstream_DB "db"

#define MT_ obs_module_text

// Audio packet info
struct cleanstream_audio_info {
  uint32_t frames;
  uint64_t timestamp;
};

struct cleanstream_data {
  obs_source_t *context;
  size_t channels;
  float multiple;

  // How many input frames (in input sample rate) are needed for the next whisper frame
  size_t frames;
  // How many frames are needed to overlap with the next whisper frame
  size_t overlap_frames;

  /* PCM buffers */
  float *copy_buffers[MAX_PREPROC_CHANNELS];
  struct circlebuf info_buffer;
  struct circlebuf input_buffers[MAX_PREPROC_CHANNELS];

  /* Resampler */
  audio_resampler_t *resampler;
  audio_resampler_t *resampler_back;

  struct whisper_context *whisper_context;
  whisper_full_params whisper_params;

  // Use std for thread and mutex
  std::thread whisper_thread;
};

std::mutex whisper_buf_mutex;
std::mutex whisper_ctx_mutex;

void high_pass_filter(float *pcmf32, size_t pcm32f_size, float cutoff, float sample_rate)
{
  const float rc = 1.0f / (2.0f * M_PI * cutoff);
  const float dt = 1.0f / sample_rate;
  const float alpha = dt / (rc + dt);

  float y = pcmf32[0];

  for (size_t i = 1; i < pcm32f_size; i++) {
    y = alpha * (y + pcmf32[i] - pcmf32[i - 1]);
    pcmf32[i] = y;
  }
}

// VAD (voice activity detection), return true if speech detected
bool vad_simple(float *pcmf32, size_t pcm32f_size, int sample_rate, int last_ms, float vad_thold,
                float freq_thold, bool verbose)
{
  const int n_samples = pcm32f_size;
  const int n_samples_last = (sample_rate * last_ms) / 1000;

  if (n_samples_last >= n_samples) {
    // not enough samples - assume no speech
    return false;
  }

  if (freq_thold > 0.0f) {
    high_pass_filter(pcmf32, pcm32f_size, freq_thold, sample_rate);
  }

  float energy_all = 0.0f;

  for (int i = 0; i < n_samples; i++) {
    energy_all += fabsf(pcmf32[i]);
  }

  energy_all /= n_samples;

  if (verbose) {
    blog(LOG_INFO, "%s: energy_all: %f, vad_thold: %f, freq_thold: %f\n",
            __func__, energy_all, vad_thold, freq_thold);
  }

  if (energy_all < vad_thold) {
    return false;
  }

  return true;
}

static const char *cleanstream_name(void *unused)
{
  UNUSED_PARAMETER(unused);
  return MT_("CleanStreamAudioFilter");
}

static void cleanstream_destroy(void *data)
{
  struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);

  {
    std::lock_guard<std::mutex> lock(whisper_ctx_mutex);
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
    std::lock_guard<std::mutex> lock(whisper_buf_mutex);
    bfree(gf->copy_buffers[0]);
    for (size_t i = 0; i < gf->channels; i++) {
      circlebuf_free(&gf->input_buffers[i]);
    }
  }

  bfree(gf);
}

static inline enum speaker_layout convert_speaker_layout(uint8_t channels)
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

static void process_audio_from_buffer(struct cleanstream_data *gf);

static void whisper_loop(void *data)
{
  struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);
  const size_t segment_size = gf->frames * sizeof(float);

  // Thread main loop
  while (true) {
    {
      std::lock_guard<std::mutex> lock(whisper_ctx_mutex);
      if (gf->whisper_context == nullptr) {
        warn("whisper context is null, exiting thread");
        return;
      }
    }

    // Check if we have enough data to process
    while (gf->input_buffers[0].size >= segment_size) {
      debug("found %d bytes, %d frames in input buffer, need >= %d, processing",
            (int)(gf->input_buffers[0].size), (int)(gf->input_buffers[0].size / sizeof(float)),
            (int)segment_size);

      // Process the audio. This will also remove the processed data from the input buffer.
      // Mutex is locked inside process_audio_from_buffer.
      process_audio_from_buffer(gf);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

static void cleanstream_update(void *data, obs_data_t *s)
{
  struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);
  double val = obs_data_get_double(s, S_cleanstream_DB);
  // Get the number of channels for the input source
  gf->channels = audio_output_get_channels(obs_get_audio());
  gf->multiple = db_to_mul((float)val);

  uint32_t sample_rate = audio_output_get_sample_rate(obs_get_audio());
  gf->frames = (size_t)(sample_rate / (1000.0f / BUFFER_SIZE_MSEC));
  gf->overlap_frames = (size_t)(sample_rate / (1000.0f / OVERLAP_SIZE_MSEC));
  info("cleanstream_update. channels %d, frames %d, sample_rate %d", (int)gf->channels,
       (int)gf->frames, sample_rate);

  struct resample_info src, dst;
  src.samples_per_sec = sample_rate;
  src.format = AUDIO_FORMAT_FLOAT_PLANAR;
  src.speakers = convert_speaker_layout((uint8_t)gf->channels);

  dst.samples_per_sec = WHISPER_SAMPLE_RATE;
  dst.format = AUDIO_FORMAT_FLOAT_PLANAR;
  dst.speakers = convert_speaker_layout((uint8_t)1);

  gf->resampler = audio_resampler_create(&dst, &src);
  gf->resampler_back = audio_resampler_create(&src, &dst);

  // allocate buffers
  gf->copy_buffers[0] = static_cast<float *>(bmalloc(gf->frames * gf->channels * sizeof(float)));
  for (size_t c = 1; c < gf->channels; c++) {
    gf->copy_buffers[c] = gf->copy_buffers[c - 1] + gf->frames;
  }
  for (size_t c = 0; c < gf->channels; c++) {
    circlebuf_reserve(&gf->input_buffers[c], gf->frames * sizeof(float));
  }

  // start the thread
  gf->whisper_thread = std::thread(whisper_loop, gf);
}

static void *cleanstream_create(obs_data_t *settings, obs_source_t *filter)
{
  struct cleanstream_data *gf =
    static_cast<struct cleanstream_data *>(bmalloc(sizeof(struct cleanstream_data)));
  gf->context = filter;
  gf->whisper_context = whisper_init_from_file(obs_module_file("models/ggml-tiny.en.bin"));
  info("cleanstream_create");

  gf->whisper_params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
  gf->whisper_params.n_threads = std::min(8, (int32_t)std::thread::hardware_concurrency());
  gf->whisper_params.duration_ms = BUFFER_SIZE_MSEC;
  gf->whisper_params.initial_prompt = "hmm, mm, mhm, mmm. uhm. Uh, um. Uhh, Umm. ehm, uuuh. Ahh, ahm.";
  gf->whisper_params.print_progress = false;
  gf->whisper_params.print_realtime = false;
  gf->whisper_params.token_timestamps = false;
  gf->whisper_params.single_segment = true;
  gf->whisper_params.suppress_non_speech_tokens = false;
  gf->whisper_params.suppress_blank = false;
  gf->whisper_params.max_tokens = 4;

  cleanstream_update(gf, settings);
  return gf;
}

static std::string to_timestamp(int64_t t)
{
  int64_t sec = t / 100;
  int64_t msec = t - sec * 100;
  int64_t min = sec / 60;
  sec = sec - min * 60;

  char buf[32];
  snprintf(buf, sizeof(buf), "%02d:%02d.%03d", (int)min, (int)sec, (int)msec);

  return std::string(buf);
}

static void run_whisper_inference(struct cleanstream_data *gf, const float *pcm32f_data,
                                  size_t pcm32f_size)
{
  debug("%s: processing %d samples, %.3f sec), %d threads", __func__, int(pcm32f_size),
        float(pcm32f_size) / WHISPER_SAMPLE_RATE, gf->whisper_params.n_threads);

  std::lock_guard<std::mutex> lock(whisper_ctx_mutex);
  // run the inference
  if (whisper_full(gf->whisper_context, gf->whisper_params, pcm32f_data, (int)pcm32f_size) != 0) {
    warn("failed to process audio");
  } else {
    const int n_segments = whisper_full_n_segments(gf->whisper_context);
    for (int n_segment = 0; n_segment < n_segments; ++n_segment) {
      const char *text = whisper_full_get_segment_text(gf->whisper_context, n_segment);
      const int64_t t0 = whisper_full_get_segment_t0(gf->whisper_context, n_segment);
      const int64_t t1 = whisper_full_get_segment_t1(gf->whisper_context, n_segment);

      info("[%s --> %s]  %s", to_timestamp(t0).c_str(), to_timestamp(t1).c_str(), text);

      // const int n = whisper_full_n_tokens(gf->whisper_context, n_segment);
      // for (int j = 0; j < n; ++j) {
      //   const whisper_token_data token =
      //     whisper_full_get_token_data(gf->whisper_context, n_segment, j);
      //   info("%d -> %d:  %s", (int)token.t0, (int)token.t1,
      //        whisper_token_to_str(gf->whisper_context, token.id));
      // }
    }
  }
}

static void process_audio_from_buffer(struct cleanstream_data *gf)
{
  {
    // scoped lock the buffer mutex
    std::lock_guard<std::mutex> lock(whisper_buf_mutex);
    /* Pop from input circlebuf */
    for (size_t c = 0; c < gf->channels; c++) {
      // move overlap frames to the beginning of copy_buffers[c]
      memcpy(gf->copy_buffers[c], gf->copy_buffers[c] + gf->frames - gf->overlap_frames,
             gf->overlap_frames * sizeof(float));
      // copy new data to the end of copy_buffers[c]
      circlebuf_pop_front(&gf->input_buffers[c], gf->copy_buffers[c] + gf->overlap_frames,
                          (gf->frames - gf->overlap_frames) * sizeof(float));
    }
  }

  // time the audio processing
  auto start = std::chrono::high_resolution_clock::now();

  // resample to 16kHz
  float *output[MAX_PREPROC_CHANNELS];
  uint32_t out_frames;
  uint64_t ts_offset;
  audio_resampler_resample(gf->resampler, (uint8_t **)output, &out_frames, &ts_offset,
                           (const uint8_t **)gf->copy_buffers, (uint32_t)gf->frames);

  debug("%d channels, %d frames, %f ms", (int)gf->channels, (int)out_frames,
        (float)out_frames / WHISPER_SAMPLE_RATE * 1000.0f);

  if (::vad_simple(output[0], out_frames, WHISPER_SAMPLE_RATE, 300, VAD_THOLD, FREQ_THOLD,
                    true)) {
    // run the inference, this is a long blocking call
    run_whisper_inference(gf, output[0], out_frames);
  } else {
    info("silence detected, skipping inference");
  }

  // end of timer
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  info("audio processing took %d ms", (int)duration);

  if (duration > BUFFER_SIZE_MSEC) {
    std::lock_guard<std::mutex> lock(whisper_buf_mutex);
    warn("audio processing took %d ms, clearing the buffer", (int)duration);
    // clear the buffer
    for (size_t c = 0; c < gf->channels; c++) {
      circlebuf_pop_front(&gf->input_buffers[c], NULL, gf->input_buffers[c].size);
    }
  }
}

static struct obs_audio_data *cleanstream_filter_audio(void *data, struct obs_audio_data *audio)
{
  if (!audio) {
    return nullptr;
  }
  if (data == nullptr) {
    return audio;
  }

  struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);

  /* -----------------------------------------------
	 * push audio packet info (timestamp/frame count) to info circlebuf */
  // struct cleanstream_audio_info info = {0};
  // info.frames = audio->frames;       // number of frames in this packet
  // info.timestamp = audio->timestamp; // timestamp of this packet
  // circlebuf_push_back(&gf->info_buffer, &info, sizeof(info));

  /* -----------------------------------------------
	 * push back current audio data to input circlebuf */
  std::lock_guard<std::mutex> lock(whisper_buf_mutex); // scoped lock
  for (size_t c = 0; c < gf->channels; c++) {
    circlebuf_push_back(&gf->input_buffers[c], audio->data[c], audio->frames * sizeof(float));
  }

  return audio;
}

static void cleanstream_defaults(obs_data_t *s)
{
  obs_data_set_default_double(s, S_cleanstream_DB, 0.0f);
}

static obs_properties_t *cleanstream_properties(void *data)
{
  obs_properties_t *ppts = obs_properties_create();

  obs_property_t *p =
    obs_properties_add_float_slider(ppts, S_cleanstream_DB, MT_("gain"), -30.0, 30.0, 0.1);
  obs_property_float_set_suffix(p, " dB");

  UNUSED_PARAMETER(data);
  return ppts;
}

struct obs_source_info my_audio_filter_info = {
  .id = "my_audio_filter",
  .type = OBS_SOURCE_TYPE_FILTER,
  .output_flags = OBS_SOURCE_AUDIO,
  .get_name = cleanstream_name,
  .create = cleanstream_create,
  .destroy = cleanstream_destroy,
  .get_defaults = cleanstream_defaults,
  .get_properties = cleanstream_properties,
  .update = cleanstream_update,
  .filter_audio = cleanstream_filter_audio,
};
