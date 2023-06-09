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

#define MAX_PREPROC_CHANNELS 8

// 500ms buffer size
#define BUFFER_SIZE_MSEC 1010
// at 16Khz, 1010ms is 16160 frames
#define WHISPER_FRAME_SIZE 16160

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

  // How many input frames are needed for the next whisper frame
  size_t frames;

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
  std::mutex whisper_buf_mutex;
};

static const char *cleanstream_name(void *unused)
{
  UNUSED_PARAMETER(unused);
  return MT_("CleanStreamAudioFilter");
}

static void cleanstream_destroy(void *data)
{
  struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);

  // join the thread
  gf->whisper_thread.join();  

  if (gf->resampler) {
    audio_resampler_destroy(gf->resampler);
    audio_resampler_destroy(gf->resampler_back);
  }
  bfree(gf->copy_buffers[0]);
  for (size_t i = 0; i < gf->channels; i++) {
    circlebuf_free(&gf->input_buffers[i]);
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

static void whisper_loop(void* data) {
  struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);
  const size_t segment_size = gf->frames * sizeof(float);

  // Thread main loop
  while (true) {
    // Check if we have enough data to process
    while (gf->input_buffers[0].size >= segment_size) {
      info("found %d bytes, %d frames in input buffer, need >= %d, processing", (int)(gf->input_buffers[0].size), 
        (int)(gf->input_buffers[0].size / sizeof(float)), (int)segment_size);

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
  info("cleanstream_update. channels %d, frames %d, sample_rate %d", gf->channels, gf->frames, sample_rate);

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
  gf->whisper_params.initial_prompt = "uh, um, uhh, umm, so, like. ";
  gf->whisper_params.print_progress   = false;
  gf->whisper_params.print_realtime   = false;

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
  info("%s: processing %d samples, %.3f sec), %d threads", __func__, int(pcm32f_size),
       float(pcm32f_size) / WHISPER_SAMPLE_RATE, gf->whisper_params.n_threads);

  // run the inference
  if (whisper_full(gf->whisper_context, gf->whisper_params, pcm32f_data, (int)pcm32f_size) != 0) {
    warn("failed to process audio");
  } else {
    const int n_segments = whisper_full_n_segments(gf->whisper_context);
    info("n_segments: %d", n_segments);
    for (int i = 0; i < n_segments; ++i) {
      const char *text = whisper_full_get_segment_text(gf->whisper_context, i);
      const int64_t t0 = whisper_full_get_segment_t0(gf->whisper_context, i);
      const int64_t t1 = whisper_full_get_segment_t1(gf->whisper_context, i);

      info("[%s --> %s]  %s", to_timestamp(t0).c_str(), to_timestamp(t1).c_str(), text);
    }
  }
}

static void process_audio_from_buffer(struct cleanstream_data *gf)
{
  // time the audio processing
  auto start = std::chrono::high_resolution_clock::now();

  // lock the buffer mutex
  gf->whisper_buf_mutex.lock();
  /* Pop from input circlebuf */
  for (size_t c = 0; c < gf->channels; c++) {
    circlebuf_pop_front(&gf->input_buffers[c], gf->copy_buffers[c], gf->frames * sizeof(float));
  }
  // unlock the buffer mutex
  gf->whisper_buf_mutex.unlock();

  // resample to 16kHz
  float *output[MAX_PREPROC_CHANNELS];
  uint32_t out_frames;
  uint64_t ts_offset;
  audio_resampler_resample(gf->resampler, (uint8_t **)output, &out_frames, &ts_offset,
                           (const uint8_t **)gf->copy_buffers, (uint32_t)gf->frames);

  info("%d channels, %d frames, %f ms", (int)gf->channels, (int)out_frames,
       (float)out_frames / WHISPER_SAMPLE_RATE * 1000.0f);

  // run the inference
  run_whisper_inference(gf, output[0], out_frames);

  // end of timer
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  info("audio processing took %d ms", (int)duration);
}

static struct obs_audio_data *cleanstream_filter_audio(void *data, struct obs_audio_data *audio)
{
  UNUSED_PARAMETER(data);
  struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);

  /* -----------------------------------------------
	 * push audio packet info (timestamp/frame count) to info circlebuf */
  // struct cleanstream_audio_info info = {0};
  // info.frames = audio->frames;       // number of frames in this packet
  // info.timestamp = audio->timestamp; // timestamp of this packet
  // circlebuf_push_back(&gf->info_buffer, &info, sizeof(info));

  /* -----------------------------------------------
	 * push back current audio data to input circlebuf */
  // lock the buffer mutex
  gf->whisper_buf_mutex.lock();
  for (size_t c = 0; c < gf->channels; c++) {
    circlebuf_push_back(&gf->input_buffers[c], audio->data[c], audio->frames * sizeof(float));
  }
  // unlock the buffer mutex
  gf->whisper_buf_mutex.unlock();

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
