"""Camera layout helpers for logical top/side processing profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from model_service.core.config import config

LEGACY_TOP_SIDE = "legacy_top_side"
DUAL_TOP_PROXY = "dual_top_proxy"
VALID_CAMERA_LAYOUTS = {LEGACY_TOP_SIDE, DUAL_TOP_PROXY}


@dataclass(frozen=True)
class CameraRole:
    """Logical camera role mapped to physical placement and processing profile."""

    logical_role: str
    physical_role: str
    processing_profile: str


def normalize_camera_layout(value: str | None) -> str:
    """Normalize camera layout names accepted from settings/env."""

    normalized = (value or LEGACY_TOP_SIDE).strip().lower()
    if normalized not in VALID_CAMERA_LAYOUTS:
        valid = ", ".join(sorted(VALID_CAMERA_LAYOUTS))
        raise ValueError(f"Invalid camera layout: {value}. Expected one of: {valid}")
    return normalized


def camera_roles_for_layout(layout: str | None = None) -> dict[str, CameraRole]:
    """Return logical top/side roles for the active physical camera layout."""

    selected_layout = normalize_camera_layout(layout or config.vision.camera_layout)
    if selected_layout == DUAL_TOP_PROXY:
        return {
            "top": CameraRole(
                logical_role="top",
                physical_role="top_middle",
                processing_profile="top",
            ),
            "side": CameraRole(
                logical_role="side",
                physical_role="top_side",
                processing_profile="top",
            ),
        }

    return {
        "top": CameraRole(
            logical_role="top",
            physical_role="top",
            processing_profile="top",
        ),
        "side": CameraRole(
            logical_role="side",
            physical_role="side",
            processing_profile="side",
        ),
    }


def camera_roles_payload(layout: str | None = None) -> dict[str, dict[str, str]]:
    """Serialize active camera roles for traces and diagnostics."""

    return {
        logical_role: asdict(role)
        for logical_role, role in camera_roles_for_layout(layout).items()
    }
