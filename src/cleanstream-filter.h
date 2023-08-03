#include <obs-module.h>

#ifdef __cplusplus
extern "C" {
#endif

void cleanstream_activate(void *data);
void *cleanstream_create(obs_data_t *settings, obs_source_t *filter);
void cleanstream_update(void *data, obs_data_t *s);
void cleanstream_destroy(void *data);
const char *cleanstream_name(void *unused);
struct obs_audio_data *cleanstream_filter_audio(void *data, struct obs_audio_data *audio);
void cleanstream_deactivate(void *data);
void cleanstream_defaults(obs_data_t *s);
obs_properties_t *cleanstream_properties(void *data);

#ifdef __cplusplus
}
#endif
