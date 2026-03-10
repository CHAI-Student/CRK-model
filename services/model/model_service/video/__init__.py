"""Video processing exports with lazy loading."""

from importlib import import_module

_EXPORTS = {
    "StreamingFrameExtractor": ("model_service.video.frame_extractor", "StreamingFrameExtractor"),
    "CV2FrameExtractor": ("model_service.video.frame_extractor", "CV2FrameExtractor"),
    "create_frame_extractor": ("model_service.video.frame_extractor", "create_frame_extractor"),
    "VoteCount": ("model_service.video.voting_ensemble", "VoteCount"),
    "VoteResult": ("model_service.video.voting_ensemble", "VoteResult"),
    "VotingEnsemble": ("model_service.video.voting_ensemble", "VotingEnsemble"),
    "VideoProcessor": ("model_service.video.video_processor", "VideoProcessor"),
}

__all__ = [
    "StreamingFrameExtractor",
    "CV2FrameExtractor",
    "create_frame_extractor",
    "VoteCount",
    "VoteResult",
    "VotingEnsemble",
    "VideoProcessor",
]


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
