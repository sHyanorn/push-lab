"""Column names and types of every processed table.

Converters read raw files with these types and synthetic fixtures are generated from them,
so the two cannot drift apart.
"""

import pyarrow as pa

Schema = dict[str, pa.DataType]

_INT = pa.int64()
_FLOAT = pa.float64()
_STR = pa.string()

KUAIRAND_LOG: Schema = {
    "user_id": _INT,
    "video_id": _INT,
    "date": _INT,
    "hourmin": _INT,
    "time_ms": _INT,
    "is_click": _INT,
    "is_like": _INT,
    "is_follow": _INT,
    "is_comment": _INT,
    "is_forward": _INT,
    "is_hate": _INT,
    "long_view": _INT,
    "play_time_ms": _INT,
    "duration_ms": _INT,
    "profile_stay_time": _INT,
    "comment_stay_time": _INT,
    "is_profile_enter": _INT,
    "is_rand": _INT,
    "tab": _INT,
}

KUAIRAND_USERS: Schema = {
    "user_id": _INT,
    "user_active_degree": _STR,
    "is_lowactive_period": _INT,
    "is_live_streamer": _INT,
    "is_video_author": _INT,
    "follow_user_num": _INT,
    "follow_user_num_range": _STR,
    "fans_user_num": _INT,
    "fans_user_num_range": _STR,
    "friend_user_num": _INT,
    "friend_user_num_range": _STR,
    "register_days": _INT,
    "register_days_range": _STR,
    # One-hot ids; some are missing for part of the users, hence float
    **{f"onehot_feat{i}": _FLOAT for i in range(18)},
}

KUAIRAND_VIDEOS_BASIC: Schema = {
    "video_id": _INT,
    "author_id": _INT,
    "video_type": _STR,
    "upload_dt": _STR,
    "upload_type": _STR,
    "visible_status": _FLOAT,
    "video_duration": _FLOAT,
    "server_width": _FLOAT,
    "server_height": _FLOAT,
    "music_id": _INT,
    "music_type": _FLOAT,
    # Comma-separated anonymous category ids
    "tag": _STR,
}

_VIDEO_STATISTIC_COUNTS = [
    "show_cnt",
    "show_user_num",
    "play_cnt",
    "play_user_num",
    "play_duration",
    "complete_play_cnt",
    "complete_play_user_num",
    "valid_play_cnt",
    "valid_play_user_num",
    "long_time_play_cnt",
    "long_time_play_user_num",
    "short_time_play_cnt",
    "short_time_play_user_num",
    "play_progress",
    "comment_stay_duration",
    "like_cnt",
    "like_user_num",
    "click_like_cnt",
    "double_click_cnt",
    "cancel_like_cnt",
    "cancel_like_user_num",
    "comment_cnt",
    "comment_user_num",
    "direct_comment_cnt",
    "reply_comment_cnt",
    "delete_comment_cnt",
    "delete_comment_user_num",
    "comment_like_cnt",
    "comment_like_user_num",
    "follow_cnt",
    "follow_user_num",
    "cancel_follow_cnt",
    "cancel_follow_user_num",
    "share_cnt",
    "share_user_num",
    "download_cnt",
    "download_user_num",
    "report_cnt",
    "report_user_num",
    "reduce_similar_cnt",
    "reduce_similar_user_num",
    "collect_cnt",
    "collect_user_num",
    "cancel_collect_cnt",
    "cancel_collect_user_num",
    "direct_comment_user_num",
    "reply_comment_user_num",
    "share_all_cnt",
    "share_all_user_num",
    "outsite_share_all_cnt",
]

KUAIRAND_VIDEOS_STATISTIC: Schema = {
    "video_id": _INT,
    "counts": _INT,
    **dict.fromkeys(_VIDEO_STATISTIC_COUNTS, _FLOAT),
}


HILLSTROM: Schema = {
    "recency": _INT,
    "history_segment": _STR,
    "history": _FLOAT,
    "mens": _INT,
    "womens": _INT,
    "zip_code": _STR,
    "newbie": _INT,
    "channel": _STR,
    "segment": _STR,
    "visit": _INT,
    "conversion": _INT,
    "spend": _FLOAT,
}

# Processed Hillstrom adds the treatment indicator for the chosen contrast
HILLSTROM_PROCESSED: Schema = {**HILLSTROM, "treatment": _INT}

_FLAG = pa.int8()

CRITEO: Schema = {
    **{f"f{i}": _FLOAT for i in range(12)},
    "treatment": _FLAG,
    "conversion": _FLAG,
    "visit": _FLAG,
    "exposure": _FLAG,
}

# Processed Criteo adds a seeded training-sample flag (D004)
CRITEO_PROCESSED: Schema = {**CRITEO, "train": pa.bool_()}


OBD_MEN_ACTIONS = 34

# The CSV files start with an unnamed pandas index column
OBD_LOG_RAW: Schema = {
    "": _INT,
    "timestamp": _STR,
    "item_id": _INT,
    "position": _INT,
    "click": _INT,
    "propensity_score": _FLOAT,
    **{f"user_feature_{i}": _STR for i in range(4)},
    **{f"user-item_affinity_{i}": _FLOAT for i in range(OBD_MEN_ACTIONS)},
}

OBD_LOG: Schema = {name.replace("-", "_"): dtype for name, dtype in OBD_LOG_RAW.items() if name}

OBD_ITEMS_RAW: Schema = {
    "": _INT,
    "item_id": _INT,
    "item_feature_0": _FLOAT,
    **{f"item_feature_{i}": _STR for i in range(1, 4)},
}

OBD_ITEMS: Schema = {name: dtype for name, dtype in OBD_ITEMS_RAW.items() if name}


def to_arrow(schema: Schema) -> pa.Schema:
    return pa.schema(list(schema.items()))
