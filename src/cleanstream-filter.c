#include "cleanstream-filter.h"

struct obs_source_info cleanstream_filter_info = {
	.id = "cleanstream_audio_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_AUDIO,
	.get_name = cleanstream_name,
	.create = cleanstream_create,
	.destroy = cleanstream_destroy,
	.get_defaults = cleanstream_defaults,
	.get_properties = cleanstream_properties,
	.update = cleanstream_update,
	.activate = cleanstream_activate,
	.deactivate = cleanstream_deactivate,
	.filter_audio = cleanstream_filter_audio,
};
