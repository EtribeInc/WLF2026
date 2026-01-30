#!/usr/bin/env python3
"""
feet_xy_osc_femto_bolt_validated.py

Purpose
-------
A "feet XY -> OSC" script that keeps the proven Orbbec connection + depth-profile selection logic
from femto_bolt_depth_blobs_spores.py, but implements the on-start background subtraction + blob
centroid -> floor XY projection + fixed-size OSC output behavior from feet_xy_osc_femto_bolt.py.

Key features
------------
- Uses OrbbecDepthCamera + DepthStreamSpec (pyorbbecsdk Pipeline-based) from femto_bolt_depth_blobs_spores.py
- Startup background capture (median) with hotkey to re-capture ('b')
- Foreground mask from depth-background difference (absolute diff threshold)
- Connected-components blobs -> centroids
- Projects centroids to floor XY (meters) using pinhole intrinsics
- Publishes fixed-size OSC payloads each frame (max_blobs, padding)

Dependencies
------------
pip install numpy opencv-python python-osc
Orbbec: pyorbbecsdk installed and working (as in femto_bolt_depth_blobs_spores.py)

Example
-------
python feet_xy_osc_femto_bolt_validated.py --preset femto_bolt_nfov_unbinned --device-index 0 \
  --osc-enabled --osc-host 127.0.0.1 --osc-port 9000 --visualize
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2
from pythonosc.udp_client import SimpleUDPClient

# Import validated capture + UI + blob detection from your existing script.
# This file must live in the same project folder (or PYTHONPATH) as femto_bolt_depth_blobs_spores.py.
from femto_bolt_depth_blobs_spores import (  # type: ignore
    DepthStreamSpec,
    OrbbecDepthCamera,
    TrackbarUI,
    ForegroundBlobDetector,
    setup_logging,
)


# -----------------------------
# Presets / intrinsics
# -----------------------------

@dataclass(frozen=True)
class Preset:
    name: str
    width: int
    height: int
    fps: int
    # "Y16" implied by OrbbecDepthCamera validation
    hfov_deg: float
    vfov_deg: float


def _intrinsics_from_fov(w: int, h: int, hfov_deg: float, vfov_deg: float) -> Tuple[float, float, float, float]:
    """
    Pinhole approx:
      fx = (w/2)/tan(HFOV/2)
      fy = (h/2)/tan(VFOV/2)
      cx = (w-1)/2
      cy = (h-1)/2
    """
    hfov = math.radians(hfov_deg)
    vfov = math.radians(vfov_deg)
    fx = (w / 2.0) / math.tan(hfov / 2.0)
    fy = (h / 2.0) / math.tan(vfov / 2.0)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    return float(fx), float(fy), float(cx), float(cy)


def make_presets() -> dict:
    # Matches the "quick-n-dirty" preset from feet_xy_osc_femto_bolt.py
    return {
        "femto_bolt_nfov_unbinned": Preset(
            name="femto_bolt_nfov_unbinned",
            width=640,
            height=576,
            fps=30,
            hfov_deg=75.0,
            vfov_deg=65.0,
        ),
    }


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


# -----------------------------
# Background capture
# -----------------------------

def _nanmedian_u16(frames_u16: List[np.ndarray]) -> np.ndarray:
    """
    Median background depth map (uint16), treating 0 as invalid.

    Returns uint16 (0 where insufficient valid samples).
    """
    if not frames_u16:
        raise ValueError("No frames provided for background capture.")

    stack = np.stack(frames_u16, axis=0).astype(np.float32)  # (N,H,W)
    stack[stack == 0] = np.nan
    bg = np.nanmedian(stack, axis=0)
    bg = np.where(np.isfinite(bg), bg, 0.0).astype(np.uint16)
    return bg


# -----------------------------
# Foreground + blobs + XY projection
# -----------------------------

def make_valid_mask(depth_u16: np.ndarray, min_depth_mm: int, max_depth_mm: int) -> np.ndarray:
    valid = (depth_u16 != 0) & (depth_u16 >= min_depth_mm) & (depth_u16 <= max_depth_mm)
    return valid


def foreground_mask_from_bg_absdiff(
    depth_u16: np.ndarray,
    bg_u16: np.ndarray,
    valid_mask: np.ndarray,
    diff_thresh_mm: int,
) -> np.ndarray:
    """
    Foreground if |depth - bg| >= diff_thresh_mm (on valid pixels).
    Returns uint8 mask 0/255.
    """
    if bg_u16 is None:
        fg = valid_mask
    else:
        diff = np.abs(depth_u16.astype(np.int32) - bg_u16.astype(np.int32)).astype(np.int32)
        fg = valid_mask & (diff >= int(diff_thresh_mm))
    return (fg.astype(np.uint8) * 255)


def clean_mask(mask_u8: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    out = mask_u8
    if open_k and open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_k and close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out


def pixel_to_floor_xy_m(u: float, v: float, intr: Intrinsics, floor_z_m: float) -> Tuple[float, float]:
    x = (u - intr.cx) / intr.fx * floor_z_m
    y = (v - intr.cy) / intr.fy * floor_z_m
    return float(x), float(y)


def robust_floor_z_m_from_bg(bg_u16: np.ndarray, depth_scale_m_per_unit: float) -> float:
    d = bg_u16.astype(np.float32) * float(depth_scale_m_per_unit)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size < 1000:
        return float("nan")
    return float(np.median(d))


# -----------------------------
# OSC
# -----------------------------

def osc_send_blobs_fixed(
    client: SimpleUDPClient,
    frame_idx: int,
    detections: List[dict],
    max_blobs: int,
    pad_value: float,
    addr_frame: str,
    addr_blobs: str,
) -> None:
    """
    Fixed-size OSC payload.

    /wlf/frame: [frame_idx, num_blobs_used]
    /wlf/blobs: [frame_idx, num_blobs_used, x0, y0, ..., xN-1, yN-1]
    where N=max_blobs, padded with pad_value (default NaN).
    """
    detections = sorted(detections, key=lambda d: d["area_px"], reverse=True)[:max_blobs]
    n = len(detections)

    client.send_message(addr_frame, [int(frame_idx), int(n)])

    payload = [int(frame_idx), int(n)]
    for det in detections:
        payload.append(float(det["x_m"]))
        payload.append(float(det["y_m"]))

    for _ in range(max_blobs - n):
        payload.append(float(pad_value))
        payload.append(float(pad_value))

    client.send_message(addr_blobs, payload)


# -----------------------------
# App
# -----------------------------

class App:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.log = logging.getLogger("feet_xy_osc")

        # Depth capture (validated in femto_bolt_depth_blobs_spores.py)
        self.cam = OrbbecDepthCamera(
            spec=DepthStreamSpec(width=args.width, height=args.height, fps=args.fps),
            timeout_ms=args.timeout_ms,
            device_index=args.device_index,
        )
        self.cam.open()

        self.detector = ForegroundBlobDetector()

        # Background
        self.bg_depth_u16: Optional[np.ndarray] = None
        self.floor_z_m: float = float("nan")

        # Intrinsics
        self.intr = Intrinsics(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)

        # Windows / UI
        self.visualize = bool(args.visualize)
        self.depth_win = "Depth / BG / Mask / Blobs"
        self.ui_win = "Controls"

        if self.visualize:
            cv2.namedWindow(self.depth_win, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self.ui_win, cv2.WINDOW_NORMAL)
            self.ui = TrackbarUI(self.ui_win)

            # Defaults are intentionally conservative; tune live.
            self.ui.add("MinDepth_mm", init=args.min_depth_mm, maxv=10000)
            self.ui.add("MaxDepth_mm", init=args.max_depth_mm, maxv=10000)
            self.ui.add("DiffThresh_mm", init=args.diff_thresh_mm, maxv=2000)
            self.ui.add("MorphOpen", init=args.morph_open, maxv=25)
            self.ui.add("MorphClose", init=args.morph_close, maxv=25)
            self.ui.add("MinArea_px", init=args.min_area_px, maxv=20000)
            self.ui.add("MaxArea_px", init=args.max_area_px, maxv=200000)
        else:
            self.ui = None

        # OSC
        self.osc_client: Optional[SimpleUDPClient] = None
        if args.osc_enabled:
            self.osc_client = SimpleUDPClient(args.osc_host, args.osc_port)

        # Capture background immediately (empty room)
        self.capture_background()

    def capture_background(self) -> None:
        self.log.info("Capturing background: %d frames...", self.args.bg_frames)
        frames: List[np.ndarray] = []
        t0 = time.time()
        while len(frames) < self.args.bg_frames:
            d = self.cam.read_depth()
            if d is None:
                continue
            frames.append(d)
        self.bg_depth_u16 = _nanmedian_u16(frames)
        self.floor_z_m = robust_floor_z_m_from_bg(self.bg_depth_u16, self.args.depth_scale)
        dt = time.time() - t0
        self.log.info("Background captured in %.2fs. floor_z_m=%.3f", dt, self.floor_z_m)

    def _params(self) -> Tuple[int, int, int, int, int, int]:
        """
        Returns: min_depth_mm, max_depth_mm, diff_thresh_mm, open_k, close_k, min_area/max_area from UI or CLI.
        """
        if self.ui is None:
            return (
                self.args.min_depth_mm,
                self.args.max_depth_mm,
                self.args.diff_thresh_mm,
                self.args.morph_open,
                self.args.morph_close,
                self.args.min_area_px,
                self.args.max_area_px,
            )
        return (
            self.ui.get("MinDepth_mm"),
            self.ui.get("MaxDepth_mm"),
            self.ui.get("DiffThresh_mm"),
            self.ui.get("MorphOpen"),
            self.ui.get("MorphClose"),
            self.ui.get("MinArea_px"),
            self.ui.get("MaxArea_px"),
        )

    def run(self) -> None:
        frame_idx = 0
        while True:
            depth_u16 = self.cam.read_depth()
            if depth_u16 is None:
                continue

            # Params
            min_depth_mm, max_depth_mm, diff_thresh_mm, open_k, close_k, min_area, max_area = self._params()

            valid = make_valid_mask(depth_u16, min_depth_mm=min_depth_mm, max_depth_mm=max_depth_mm)
            mask = foreground_mask_from_bg_absdiff(
                depth_u16=depth_u16,
                bg_u16=self.bg_depth_u16,
                valid_mask=valid,
                diff_thresh_mm=diff_thresh_mm,
            )
            mask = clean_mask(mask, open_k=open_k, close_k=close_k)

            blobs = self.detector.detect(mask, min_area=min_area, max_area=max_area)

            # Project to floor XY (meters)
            detections: List[dict] = []
            if np.isfinite(self.floor_z_m) and self.floor_z_m > 0:
                for b in blobs:
                    u, v = b.centroid_xy
                    x_m, y_m = pixel_to_floor_xy_m(u, v, self.intr, self.floor_z_m)
                    detections.append(
                        {
                            "u_px": float(u),
                            "v_px": float(v),
                            "area_px": int(b.area),
                            "x_m": float(x_m),
                            "y_m": float(y_m),
                        }
                    )

            # OSC
            if self.osc_client is not None:
                osc_send_blobs_fixed(
                    self.osc_client,
                    frame_idx=frame_idx,
                    detections=detections,
                    max_blobs=self.args.max_blobs,
                    pad_value=self.args.osc_pad_value,
                    addr_frame=self.args.osc_addr_frame,
                    addr_blobs=self.args.osc_addr_blobs,
                )

            # Visualize
            if self.visualize:
                # Depth visualization
                d = depth_u16.astype(np.float32) * float(self.args.depth_scale)
                d[depth_u16 == 0] = 0.0
                clip = float(self.args.clip_max_m)
                if clip > 0:
                    d = np.clip(d, 0, clip)
                    denom = clip
                else:
                    denom = max(float(np.max(d)), 1e-6)

                img = (d / denom * 255.0).astype(np.uint8)
                vis = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)

                # Draw blobs
                for det in sorted(detections, key=lambda x: x["area_px"], reverse=True)[: self.args.max_blobs]:
                    u0 = int(round(det["u_px"]))
                    v0 = int(round(det["v_px"]))
                    cv2.drawMarker(vis, (u0, v0), (255, 255, 255),
                                   markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
                    cv2.putText(
                        vis,
                        f"{det['x_m']:.2f},{det['y_m']:.2f}",
                        (u0 + 8, v0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # Composite view: left=depth, mid=mask, right=bg (if present)
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                panels = [vis, mask_bgr]
                if self.bg_depth_u16 is not None:
                    bg_m = self.bg_depth_u16.astype(np.float32) * float(self.args.depth_scale)
                    bg_m[self.bg_depth_u16 == 0] = 0.0
                    if clip > 0:
                        bg_m = np.clip(bg_m, 0, clip)
                        denom2 = clip
                    else:
                        denom2 = max(float(np.max(bg_m)), 1e-6)
                    bg_img = (bg_m / denom2 * 255.0).astype(np.uint8)
                    bg_vis = cv2.applyColorMap(bg_img, cv2.COLORMAP_TURBO)
                    panels.append(bg_vis)

                show = np.hstack(panels)
                cv2.imshow(self.depth_win, show)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("b"):
                    self.capture_background()

            frame_idx += 1

        if self.visualize:
            cv2.destroyAllWindows()


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    presets = make_presets()
    ap = argparse.ArgumentParser()

    ap.add_argument("--preset", type=str, default="femto_bolt_nfov_unbinned",
                    help=f"Named preset. Options: {', '.join(presets.keys())}")

    # Device / capture
    ap.add_argument("--device-index", type=int, default=0, help="Orbbec device index (0..N-1).")
    ap.add_argument("--timeout-ms", type=int, default=100, help="Frame wait timeout in ms.")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--fps", type=int, default=0)

    # Depth scale: Orbbec Y16 usually mm
    ap.add_argument("--depth-scale", type=float, default=0.001, help="Meters per depth unit (often 0.001).")

    # Background
    ap.add_argument("--bg-frames", type=int, default=30, help="Number of startup frames for background median.")

    # Defaults for processing (overridden by trackbars if --visualize)
    ap.add_argument("--min-depth-mm", type=int, default=200)
    ap.add_argument("--max-depth-mm", type=int, default=6000)
    ap.add_argument("--diff-thresh-mm", type=int, default=60)
    ap.add_argument("--morph-open", type=int, default=3)
    ap.add_argument("--morph-close", type=int, default=9)
    ap.add_argument("--min-area-px", type=int, default=500)
    ap.add_argument("--max-area-px", type=int, default=50000)

    # Intrinsics
    ap.add_argument("--fx", type=float, default=float("nan"))
    ap.add_argument("--fy", type=float, default=float("nan"))
    ap.add_argument("--cx", type=float, default=float("nan"))
    ap.add_argument("--cy", type=float, default=float("nan"))

    # OSC
    ap.add_argument("--osc-enabled", action="store_true", help="Enable OSC output.")
    ap.add_argument("--osc-host", type=str, default="127.0.0.1")
    ap.add_argument("--osc-port", type=int, default=9000)
    ap.add_argument("--osc-addr-frame", type=str, default="/wlf/frame")
    ap.add_argument("--osc-addr-blobs", type=str, default="/wlf/blobs")
    ap.add_argument("--max-blobs", type=int, default=8, help="Fixed maximum blob count in OSC payload.")
    ap.add_argument("--osc-pad-value", type=float, default=float("nan"), help="Pad value for unused blob slots.")

    # Viz/UI
    ap.add_argument("--visualize", action="store_true", help="Show OpenCV windows + enable trackbar tuning.")
    ap.add_argument("--clip-max-m", type=float, default=4.0, help="Depth colormap clip (meters).")

    # Logging
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    # Resolve preset -> capture spec defaults + intrinsics (unless user overrides)
    if args.preset not in presets:
        raise SystemExit(f"Unknown preset '{args.preset}'. Options: {list(presets.keys())}")
    p = presets[args.preset]

    if args.width <= 0:
        args.width = p.width
    if args.height <= 0:
        args.height = p.height
    if args.fps <= 0:
        args.fps = p.fps

    # Intrinsics from FOV if not provided
    if not np.isfinite(args.fx) or not np.isfinite(args.fy) or not np.isfinite(args.cx) or not np.isfinite(args.cy):
        fx, fy, cx, cy = _intrinsics_from_fov(args.width, args.height, p.hfov_deg, p.vfov_deg)
        if not np.isfinite(args.fx):
            args.fx = fx
        if not np.isfinite(args.fy):
            args.fy = fy
        if not np.isfinite(args.cx):
            args.cx = cx
        if not np.isfinite(args.cy):
            args.cy = cy

    return args


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    app = App(args)
    app.run()


if __name__ == "__main__":
    main()
