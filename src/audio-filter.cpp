#include <obs-module.h>
#include <media-io/audio-math.h>
#include <media-io/audio-resampler.h>
#include <util/circlebuf.h>
#include <util/darray.h>
#include <math.h>

#include <string>
#include <thread>
#include <mutex>
#include <cinttypes>
#include <algorithm>
#include <regex>

#include <whisper.h>

#define do_log(level, format, ...) \
  blog(level, "[cleanstream filter: '%s'] " format, __func__, ##__VA_ARGS__)

#define error(format, ...) do_log(LOG_ERROR, format, ##__VA_ARGS__)
#define warn(format, ...) do_log(LOG_WARNING, format, ##__VA_ARGS__)
#define info(format, ...) do_log(LOG_INFO, format, ##__VA_ARGS__)
#define debug(format, ...) do_log(LOG_DEBUG, format, ##__VA_ARGS__)

#define MAX_PREPROC_CHANNELS 2

// buffer size in msec
#define BUFFER_SIZE_MSEC 1010
// at 16Khz, 1010 msec is 16160 frames
#define WHISPER_FRAME_SIZE 16160
// overlap in msec
#define OVERLAP_SIZE_MSEC 340

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
  obs_source_t *context; // obs input source
  size_t channels;       // number of channels
  uint32_t sample_rate;  // input sample rate
  // How many input frames (in input sample rate) are needed for the next whisper frame
  size_t frames;
  // How many ms/frames are needed to overlap with the next whisper frame
  size_t overlap_frames;
  size_t overlap_ms;
  // How many frames were processed in the last whisper frame (this is dynamic)
  size_t last_num_frames;

  /* PCM buffers */
  float *copy_buffers[MAX_PREPROC_CHANNELS];
  struct circlebuf info_buffer;
  struct circlebuf info_out_buffer;
  struct circlebuf input_buffers[MAX_PREPROC_CHANNELS];
  struct circlebuf output_buffers[MAX_PREPROC_CHANNELS];

  /* Resampler */
  audio_resampler_t *resampler;
  audio_resampler_t *resampler_back;

  struct whisper_context *whisper_context;
  whisper_full_params whisper_params;

  // Use std for thread and mutex
  std::thread whisper_thread;

  /* output data */
  struct obs_audio_data output_audio;
  DARRAY(float) output_data;

  float filler_p_threshold;

  bool do_silence;
  bool vad_enabled;
  int log_level;
  std::string detect_regex;
};

std::mutex whisper_buf_mutex;
std::mutex whisper_outbuf_mutex;
std::mutex whisper_ctx_mutex;

static void whisper_loop(void *data);

void high_pass_filter(float *pcmf32, size_t pcm32f_size, float cutoff, uint32_t sample_rate)
{
  const float rc = 1.0f / (2.0f * M_PI * cutoff);
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

  energy_all /= n_samples;

  if (verbose) {
    blog(LOG_DEBUG, "%s: energy_all: %f, vad_thold: %f, freq_thold: %f\n", __func__, energy_all,
         vad_thold, freq_thold);
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
  energy_in_window /= n_samples_window;

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
  UNUSED_PARAMETER(pcm32f_size);

  // scan the buffer with a window of 50ms
  const uint64_t n_samples_window = (sample_rate * 50) / 1000;

  float first_window_energy = avg_energy_in_window(pcmf32, 0, n_samples_window);
  float last_window_energy =
    avg_energy_in_window(pcmf32, pcm32f_size - n_samples_window, n_samples_window);
  float max_energy_in_middle =
    max_energy_in_window(pcmf32, n_samples_window, pcm32f_size - n_samples_window);

  if (verbose) {
    blog(LOG_INFO, "%s: first_window_energy: %f, last_window_energy: %f, max_energy_in_middle: %f",
         __func__, first_window_energy, last_window_energy, max_energy_in_middle);
  }

  // print avg energy in all windows in sample
  for (uint64_t i = 0; i < pcm32f_size - n_samples_window; i += n_samples_window) {
    blog(LOG_INFO, "%s: avg energy_in_window %llu: %f", __func__, i,
         avg_energy_in_window(pcmf32, i, n_samples_window));
  }

  const float max_energy_thold = max_energy_in_middle * thold;
  if (first_window_energy < max_energy_thold && last_window_energy < max_energy_thold) {
    if (verbose) {
      blog(LOG_INFO, "%s: word boundary found between %llu and %llu\n", __func__, n_samples_window,
           pcm32f_size - n_samples_window);
    }
    return n_samples_window;
  }

  return 0;
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

static struct whisper_context *init_whisper_context()
{
  struct whisper_context *ctx = whisper_init_from_file(obs_module_file("models/ggml-tiny.en.bin"));
  if (ctx == nullptr) {
    error("Failed to load whisper model");
    return nullptr;
  }
  return ctx;
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

static bool run_whisper_inference(struct cleanstream_data *gf, const float *pcm32f_data,
                                  size_t pcm32f_size)
{
  do_log(gf->log_level, "%s: processing %d samples, %.3f sec, %d threads", __func__, int(pcm32f_size),
        float(pcm32f_size) / WHISPER_SAMPLE_RATE, gf->whisper_params.n_threads);

  std::lock_guard<std::mutex> lock(whisper_ctx_mutex);
  if (gf->whisper_context == nullptr) {
    warn("whisper context is null");
    return false;
  }

  // run the inference
  int whisper_full_result = -1;
  try {
    whisper_full_result =
      whisper_full(gf->whisper_context, gf->whisper_params, pcm32f_data, (int)pcm32f_size);
  } catch (const std::exception &e) {
    error("Whisper exception: %s. Filter restart is required", e.what());
    whisper_free(gf->whisper_context);
    gf->whisper_context = nullptr;
  }

  if (whisper_full_result != 0) {
    warn("failed to process audio, error %d", whisper_full_result);
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

    // if text (convert to lowercase) contains `[blank` or `uh,` or `uh...` then we have a
    // blank segment
    std::string text_lower(text);
    std::transform(text_lower.begin(), text_lower.end(), text_lower.begin(), ::tolower);
    info("[%s --> %s] (%.3f) %s", to_timestamp(t0).c_str(), to_timestamp(t1).c_str(), sentence_p,
         text_lower.c_str());

    // use a regular expression to detect filler words with a word boundary
    try {
      std::regex filler_regex(gf->detect_regex);
      if (std::regex_search(text_lower, filler_regex)) {
        return true;
      }
    } catch (const std::regex_error &e) {
      error("Regex error: %s", e.what());
    }
  }

  return false;
}

static void process_audio_from_buffer(struct cleanstream_data *gf);

static void whisper_loop(void *data)
{
  struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);
  const size_t segment_size = gf->frames * sizeof(float);

  info("starting whisper thread");

  // Thread main loop
  while (true) {
    {
      std::lock_guard<std::mutex> lock(whisper_ctx_mutex);
      if (gf->whisper_context == nullptr) {
        warn("Whisper context is null, exiting thread");
        break;
      }
    }

    // Check if we have enough data to process
    while (true) {
      size_t input_buf_size = 0;
      {
        std::lock_guard<std::mutex> lock(whisper_buf_mutex);
        input_buf_size = gf->input_buffers[0].size;
      }

      if (input_buf_size >= segment_size) {
        do_log(gf->log_level, "found %lu bytes, %lu frames in input buffer, need >= %lu, processing",
              input_buf_size, (size_t)(input_buf_size / sizeof(float)), segment_size);

        // Process the audio. This will also remove the processed data from the input buffer.
        // Mutex is locked inside process_audio_from_buffer.
        process_audio_from_buffer(gf);
      } else {
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  info("exiting whisper thread");
}

static void process_audio_from_buffer(struct cleanstream_data *gf)
{
  uint32_t num_new_frames_from_infos = 0;
  uint64_t start_timestamp = 0;

  {
    // scoped lock the buffer mutex
    std::lock_guard<std::mutex> lock(whisper_buf_mutex);

    // pop infos from the info buffer and mark the beginning timestamp from the first
    // info as the beginning timestamp of the segment
    struct cleanstream_audio_info info_from_buf = {0};
    while (gf->info_buffer.size >= sizeof(struct cleanstream_audio_info)) {
      circlebuf_pop_front(&gf->info_buffer, &info_from_buf, sizeof(struct cleanstream_audio_info));
      num_new_frames_from_infos += info_from_buf.frames;
      if (start_timestamp == 0) {
        start_timestamp = info_from_buf.timestamp;
      }
      if (num_new_frames_from_infos >= (gf->frames - gf->overlap_frames)) {
        // push the last info back into the buffer
        circlebuf_push_back(&gf->info_buffer, &info_from_buf,
                            sizeof(struct cleanstream_audio_info));
        num_new_frames_from_infos -= info_from_buf.frames;
        break;
      }
    }

    /* Pop from input circlebuf */
    for (size_t c = 0; c < gf->channels; c++) {
      if (gf->last_num_frames > 0) {
        // move overlap frames from the end of the last copy_buffers to the beginning
        memcpy(gf->copy_buffers[c], gf->copy_buffers[c] + gf->last_num_frames - gf->overlap_frames,
               gf->overlap_frames * sizeof(float));
        // copy new data to the end of copy_buffers[c]
        circlebuf_pop_front(&gf->input_buffers[c], gf->copy_buffers[c] + gf->overlap_frames,
                            num_new_frames_from_infos * sizeof(float));
      } else {
        // Very first time, just copy data to the end of copy_buffers[c]
        circlebuf_pop_front(&gf->input_buffers[c], gf->copy_buffers[c],
                            (num_new_frames_from_infos + gf->overlap_frames) * sizeof(float));
      }
    }

    gf->last_num_frames = num_new_frames_from_infos + gf->overlap_frames;
  }

  do_log(gf->log_level, "processing %d frames (%d ms), start timestamp %" PRIu64 " ", (int)gf->last_num_frames,
        (int)(gf->last_num_frames * 1000 / gf->sample_rate), start_timestamp);

  // time the audio processing
  auto start = std::chrono::high_resolution_clock::now();

  // resample to 16kHz
  float *output[MAX_PREPROC_CHANNELS];
  uint32_t out_frames;
  uint64_t ts_offset;
  audio_resampler_resample(gf->resampler, (uint8_t **)output, &out_frames, &ts_offset,
                           (const uint8_t **)gf->copy_buffers, (uint32_t)gf->last_num_frames);

  do_log(gf->log_level, "%d channels, %d frames, %f ms", (int)gf->channels, (int)out_frames,
        (float)out_frames / WHISPER_SAMPLE_RATE * 1000.0f);

  bool skipped_inference = false;

  if (gf->vad_enabled) {
    skipped_inference = !::vad_simple(output[0], out_frames, WHISPER_SAMPLE_RATE, VAD_THOLD, FREQ_THOLD, false);
  }

  if (!skipped_inference) {
    if (run_whisper_inference(gf, output[0], out_frames)) {
      // this is a filler segment, reduce the output volume

      // find first word boundary, up to 50% of the way through the segment
      // const size_t first_boundary = word_boundary_simple(gf->copy_buffers[0], num_new_frames_from_infos,
      //                                                    num_new_frames_from_infos / 2,
      //                                                    gf->sample_rate, 0.1f, true);
      const size_t first_boundary = 0;

      info("filler segment, reducing volume on frames %lu -> %u", first_boundary,
          num_new_frames_from_infos);

      if (gf->do_silence) {
        for (size_t c = 0; c < gf->channels; c++) {
          for (size_t i = first_boundary; i < num_new_frames_from_infos; i++) {
            gf->copy_buffers[c][i] = 0;
          }
        }
      }
    }
  } else {
    info("skipping inference");
  }

  {
    std::lock_guard<std::mutex> lock(whisper_outbuf_mutex);

    struct cleanstream_audio_info info_out = {0};
    info_out.frames = num_new_frames_from_infos; // number of frames in this packet
    info_out.timestamp = start_timestamp;        // timestamp of this packet
    circlebuf_push_back(&gf->info_out_buffer, &info_out, sizeof(info_out));

    for (size_t c = 0; c < gf->channels; c++) {
      circlebuf_push_back(&gf->output_buffers[c], gf->copy_buffers[c],
                          (num_new_frames_from_infos) * sizeof(float));
    }
    // log sizes of output buffers
    do_log(gf->log_level, "output info buffer size: %lu, output data buffer size bytes: %lu",
          gf->info_out_buffer.size / sizeof(struct cleanstream_audio_info),
          gf->output_buffers[0].size);
  }

  // end of timer
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  const uint32_t new_frames_from_infos_ms =
    num_new_frames_from_infos * 1000 / gf->sample_rate; // number of frames in this packet
  do_log(gf->log_level, "audio processing of %u ms new data took %d ms", new_frames_from_infos_ms, (int)duration);

  if (duration > new_frames_from_infos_ms) {
    // try to decrease overlap down to minimum of 100 ms
    gf->overlap_ms = std::max((uint64_t)gf->overlap_ms - 10, (uint64_t)100);
    gf->overlap_frames = gf->overlap_ms * gf->sample_rate / 1000;
    do_log(gf->log_level, "audio processing took too long (%d ms), reducing overlap to %lu ms", (int)duration,
          gf->overlap_ms);
  } else if (!skipped_inference) {
    // try to increase overlap up to 75% of the segment
    gf->overlap_ms =
      std::min((uint64_t)gf->overlap_ms + 10, (uint64_t)(new_frames_from_infos_ms * 0.75f));
    gf->overlap_frames = gf->overlap_ms * gf->sample_rate / 1000;
    do_log(gf->log_level, "audio processing took %d ms, increasing overlap to %lu ms", (int)duration,
          gf->overlap_ms);
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

  if (gf->whisper_context == nullptr) {
    // Whisper not initialized, just pass through
    return audio;
  }

  {
    std::lock_guard<std::mutex> lock(whisper_buf_mutex); // scoped lock
    /* -----------------------------------------------
     * push back current audio data to input circlebuf */
    for (size_t c = 0; c < gf->channels; c++) {
      circlebuf_push_back(&gf->input_buffers[c], audio->data[c], audio->frames * sizeof(float));
    }
    /* -----------------------------------------------
     * push audio packet info (timestamp/frame count) to info circlebuf */
    struct cleanstream_audio_info info = {0};
    info.frames = audio->frames;       // number of frames in this packet
    info.timestamp = audio->timestamp; // timestamp of this packet
    circlebuf_push_back(&gf->info_buffer, &info, sizeof(info));
  }

  // Check for output to play
  struct cleanstream_audio_info info_out = {0};
  {
    std::lock_guard<std::mutex> lock(whisper_outbuf_mutex); // scoped lock

    if (gf->info_out_buffer.size == 0) {
      // nothing to output
      return NULL;
    }

    // pop from output buffers to get audio packet info
    circlebuf_pop_front(&gf->info_out_buffer, &info_out, sizeof(info_out));
    do_log(gf->log_level, "output packet info: timestamp=%llu, frames=%u, bytes=%lu, ms=%u", info_out.timestamp,
          info_out.frames, gf->output_buffers[0].size, info_out.frames * 1000 / gf->sample_rate);

    // prepare output data buffer
    da_resize(gf->output_data, info_out.frames * gf->channels * sizeof(float));

    // pop from output circlebuf to audio data
    for (size_t i = 0; i < gf->channels; i++) {
      gf->output_audio.data[i] = (uint8_t *)&gf->output_data.array[i * info_out.frames];
      circlebuf_pop_front(&gf->output_buffers[i], gf->output_audio.data[i],
                          info_out.frames * sizeof(float));
    }
  }

  gf->output_audio.frames = info_out.frames;
  gf->output_audio.timestamp = info_out.timestamp;
  return &gf->output_audio;
}

static const char *cleanstream_name(void *unused)
{
  UNUSED_PARAMETER(unused);
  return MT_("CleanStreamAudioFilter");
}

static void cleanstream_destroy(void *data)
{
  struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);

  info("cleanstream_destroy");
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
    std::lock_guard<std::mutex> lockbuf(whisper_buf_mutex);
    std::lock_guard<std::mutex> lockoutbuf(whisper_outbuf_mutex);
    bfree(gf->copy_buffers[0]);
    gf->copy_buffers[0] = nullptr;
    for (size_t i = 0; i < gf->channels; i++) {
      circlebuf_free(&gf->input_buffers[i]);
      circlebuf_free(&gf->output_buffers[i]);
    }
  }
  circlebuf_free(&gf->info_buffer);
  circlebuf_free(&gf->info_out_buffer);
  da_free(gf->output_data);

  bfree(gf);
}

static void cleanstream_update(void *data, obs_data_t *s)
{
  struct cleanstream_data *gf = static_cast<struct cleanstream_data *>(data);

  gf->filler_p_threshold = (float)obs_data_get_double(s, "filler_p_threshold");
  gf->log_level = (int)obs_data_get_int(s, "log_level");
  gf->do_silence = obs_data_get_bool(s, "do_silence");
  gf->vad_enabled = obs_data_get_bool(s, "vad_enabled");
  gf->detect_regex = obs_data_get_string(s, "detect_regex");

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

static void *cleanstream_create(obs_data_t *settings, obs_source_t *filter)
{
  struct cleanstream_data *gf =
    static_cast<struct cleanstream_data *>(bmalloc(sizeof(struct cleanstream_data)));

  // Get the number of channels for the input source
  gf->channels = audio_output_get_channels(obs_get_audio());
  gf->sample_rate = audio_output_get_sample_rate(obs_get_audio());
  gf->frames = (size_t)(gf->sample_rate / (1000.0f / BUFFER_SIZE_MSEC));

  for (size_t i = 0; i < MAX_AUDIO_CHANNELS; i++) {
    gf->copy_buffers[i] = nullptr;
    circlebuf_init(&gf->input_buffers[i]);
    circlebuf_init(&gf->output_buffers[i]);
  }
  circlebuf_init(&gf->info_buffer);
  circlebuf_init(&gf->info_out_buffer);
  da_init(gf->output_data);

  // allocate copy buffers
  gf->copy_buffers[0] = static_cast<float *>(bzalloc(gf->channels * gf->frames * sizeof(float)));
  for (size_t i = 0; i < gf->channels; i++) { // set the channel pointers
    gf->copy_buffers[i] = gf->copy_buffers[0] + i * gf->frames;
  }

  gf->context = filter;
  gf->whisper_context = init_whisper_context();
  if (gf->whisper_context == nullptr) {
    error("Failed to load whisper model");
    return nullptr;
  }

  gf->whisper_params = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
  gf->whisper_params.duration_ms = BUFFER_SIZE_MSEC;

  gf->overlap_ms = OVERLAP_SIZE_MSEC;
  gf->overlap_frames = (size_t)(gf->sample_rate / (1000.0f / gf->overlap_ms));
  info("CleanStream filter: channels %d, frames %d, sample_rate %d", (int)gf->channels,
       (int)gf->frames, gf->sample_rate);

  struct resample_info src, dst;
  src.samples_per_sec = gf->sample_rate;
  src.format = AUDIO_FORMAT_FLOAT_PLANAR;
  src.speakers = convert_speaker_layout((uint8_t)gf->channels);

  dst.samples_per_sec = WHISPER_SAMPLE_RATE;
  dst.format = AUDIO_FORMAT_FLOAT_PLANAR;
  dst.speakers = convert_speaker_layout((uint8_t)1);

  gf->resampler = audio_resampler_create(&dst, &src);
  gf->resampler_back = audio_resampler_create(&src, &dst);

  gf->do_silence = obs_data_get_bool(settings, "do_silence");
  gf->vad_enabled = obs_data_get_bool(settings, "vad_enabled");
  gf->log_level = (int)obs_data_get_int(settings, "log_level");

  cleanstream_update(gf, settings);

  // start the thread
  gf->whisper_thread = std::thread(whisper_loop, gf);

  return gf;
}

static void cleanstream_defaults(obs_data_t *s)
{
  obs_data_set_default_double(s, "filler_p_threshold", 0.75);
  obs_data_set_default_bool(s, "do_silence", true);
  obs_data_set_default_bool(s, "vad_enabled", true);
  obs_data_set_default_int(s, "log_level", LOG_DEBUG);
  obs_data_set_default_string(s, "detect_regex", "\\b(uh|um|ah|eh|\\[bl)\\b");
  obs_data_set_default_string(
    s, "initial_prompt",
    "hmm, mm, mhm, mmm, uhm, Uh, um, Uhh, Umm, ehm, uuuh, Ahh, ahm, eh, Ehh, ehh,");
  obs_data_set_default_int(s, "n_threads", 8);
  obs_data_set_default_int(s, "n_max_text_ctx", 16384);
  obs_data_set_default_bool(s, "no_context", true);
  obs_data_set_default_bool(s, "single_segment", false);
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
  obs_data_set_default_bool(s, "suppress_non_speech_tokens", false);
  obs_data_set_default_double(s, "temperature", 0.0);
  obs_data_set_default_double(s, "max_initial_ts", 1.0);
  obs_data_set_default_double(s, "length_penalty", -1.0);
}

static obs_properties_t *cleanstream_properties(void *data)
{
  obs_properties_t *ppts = obs_properties_create();

  obs_properties_add_float_slider(ppts, "filler_p_threshold", "filler_p_threshold", 0.0f, 1.0f,
                                  0.05f);
  obs_properties_add_bool(ppts, "do_silence", "do_silence");
  obs_properties_add_bool(ppts, "vad_enabled", "vad_enabled");
  obs_property_t* list = obs_properties_add_list(ppts, "log_level", "log_level", OBS_COMBO_TYPE_LIST,
                          OBS_COMBO_FORMAT_INT);
  obs_property_list_add_int(list, "DEBUG", LOG_DEBUG);
  obs_property_list_add_int(list, "INFO", LOG_INFO);
  obs_property_list_add_int(list, "WARNING", LOG_WARNING);
  obs_property_list_add_int(list, "ERROR", LOG_ERROR);
  obs_properties_add_text(ppts, "detect_regex", "detect_regex", OBS_TEXT_DEFAULT);

  obs_properties_t *whisper_params_group = obs_properties_create();
  obs_properties_add_group(ppts, "whisper_params_group", "Whisper Parameters", OBS_GROUP_NORMAL,
                           whisper_params_group);

  // int n_threads;
  obs_properties_add_int_slider(whisper_params_group, "n_threads", "n_threads", 1, 8, 1);
  // int n_max_text_ctx;     // max tokens to use from past text as prompt for the decoder
  obs_properties_add_int_slider(whisper_params_group, "n_max_text_ctx", "n_max_text_ctx", 0, 16384,
                                100);
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
  obs_properties_add_float_slider(whisper_params_group, "thold_pt", "thold_pt", 0.0f, 1.0f, 0.05f);
  // float thold_ptsum;      // timestamp token sum probability threshold (~0.01)
  obs_properties_add_float_slider(whisper_params_group, "thold_ptsum", "thold_ptsum", 0.0f, 1.0f,
                                  0.05f);
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
  // bool suppress_blank;    // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L89
  obs_properties_add_bool(whisper_params_group, "suppress_blank", "suppress_blank");
  // bool suppress_non_speech_tokens; // ref: https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/tokenizer.py#L224-L253
  obs_properties_add_bool(whisper_params_group, "suppress_non_speech_tokens",
                          "suppress_non_speech_tokens");
  // float temperature;      // initial decoding temperature, ref: https://ai.stackexchange.com/a/32478
  obs_properties_add_float_slider(whisper_params_group, "temperature", "temperature", 0.0f, 1.0f,
                                  0.05f);
  // float max_initial_ts;   // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L97
  obs_properties_add_float_slider(whisper_params_group, "max_initial_ts", "max_initial_ts", 0.0f,
                                  1.0f, 0.05f);
  // float length_penalty;   // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L267
  obs_properties_add_float_slider(whisper_params_group, "length_penalty", "length_penalty", -1.0f,
                                  1.0f, 0.1f);

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
