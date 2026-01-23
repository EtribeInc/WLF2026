#!/usr/bin/env python3
"""
Femto Bolt depth -> user-captured background subtraction -> blob detection -> particle “spore field” viz.

Depth capture uses OpenCV obsensor backend:
    VideoCapture(index, cv2.CAP_OBSENSOR)
    retrieve(depth, cv2.CAP_OBSENSOR_DEPTH_MAP)

References:
- OpenCV Orbbec UVC (Astra+/Femto) support example shows CAP_OBSENSOR + CAP_OBSENSOR_DEPTH_MAP.

Author: Jeff Kranski (jeff@kranskilabs.com)
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import logging
import math
import time
from typing import List, Optional, Tuple

import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, Context
import cv2


# ---------------------------
# Logging
# ---------------------------

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------
# UI Trackbars
# ---------------------------

class TrackbarUI:
    """
    Centralized OpenCV trackbar management.
    """
    def __init__(self, window_name: str):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def add(self, name: str, init: int, maxv: int) -> None:
        cv2.createTrackbar(name, self.window_name, init, maxv, lambda _v: None)

    def get(self, name: str) -> int:
        return int(cv2.getTrackbarPos(name, self.window_name))


# ---------------------------
# Depth capture
# ---------------------------
@dataclass(frozen=True)
class DepthStreamSpec:
    width: int = 640
    height: int = 400
    fps: int = 15
    fmt: OBFormat = OBFormat.Y16


def _choose_depth_profile(profile_list, preferred, logger):
    """
    preferred: list of tuples (w, h, fmt, fps) in priority order.
    Returns a VideoStreamProfile.

    Falls back to the first available depth profile if no preferred match exists.
    """
    # Try exact matches first
    for (w, h, fmt, fps) in preferred:
        try:
            return profile_list.get_video_stream_profile(w, h, fmt, fps)
        except Exception:
            pass

    # Enumerate and log available profiles
    avail = []
    try:
        n = profile_list.get_count()
        for i in range(n):
            p = profile_list.get_profile(i).as_video_stream_profile()
            avail.append((p.get_width(), p.get_height(), p.get_format(), p.get_fps()))
    except Exception as e:
        logger.warning("Failed to enumerate stream profiles: %s", e)

    if avail:
        logger.info("Available depth profiles:")
        for (w, h, fmt, fps) in avail:
            logger.info("  %dx%d @ %d  fmt=%s", w, h, fps, fmt)

        # Choose “best” by a simple heuristic:
        # - prefer higher fps (>= preferred fps if given)
        # - prefer moderate resolution (good blob segmentation, lower bandwidth)
        # You can tweak this later.
        def score(item):
            w, h, fmt, fps = item
            # Prefer Y16 / Z16-like formats
            fmt_bonus = 1000 if str(fmt) in ("OBFormat.Y16", "OBFormat.Z16") else 0
            # Prefer 30fps, then 15fps
            fps_bonus = 200 if fps >= 30 else (100 if fps >= 15 else 0)
            # Prefer ~640x480-ish area
            area = w * h
            area_penalty = abs(area - (640 * 480)) / 1000.0
            return fmt_bonus + fps_bonus - area_penalty

        best = max(avail, key=score)
        bw, bh, bfmt, bfps = best
        logger.info("Chose depth profile: %dx%d @ %d fmt=%s", bw, bh, bfps, bfmt)
        return profile_list.get_video_stream_profile(bw, bh, bfmt, bfps)

    raise RuntimeError(
        "No usable depth profiles found (could not match preferred and could not enumerate)."
    )


class OrbbecDepthCamera:
    """
    Robust depth capture using Orbbec SDK v2 (pyorbbecsdk).

    Returns:
        depth_mm: np.ndarray shape (H, W) dtype uint16, depth in millimeters.
    """

    def __init__(
        self,
        spec: DepthStreamSpec = DepthStreamSpec(),
        timeout_ms: int = 100,
        device_index: int = 0,
    ):
        self.spec = spec
        self.timeout_ms = int(timeout_ms)
        self.device_index = int(device_index)

        self._ctx: Optional[Context] = None
        self._device = None
        self._pipeline: Optional[Pipeline] = None
        self._started = False
        self.log = logging.getLogger(__name__)

    # -----------------------------
    # Lifecycle
    # -----------------------------

    def open(self) -> None:
        if self._started:
            return

        self._ctx = Context()
        device_list = self._ctx.query_devices()

        if device_list is None or device_list.get_count() == 0:
            raise RuntimeError("No Orbbec devices detected.")

        if self.device_index >= device_list.get_count():
            raise RuntimeError(
                f"device_index={self.device_index} out of range; "
                f"{device_list.get_count()} device(s) available."
            )

        self._device = device_list.get_device_by_index(self.device_index)
        self._pipeline = Pipeline(self._device)

        config = Config()

        # Select an explicit depth profile
        profile_list = self._pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        )

        preferred = [
            # Common “safe” depth modes (you can expand once you see what the Bolt reports)
            (640, 480, OBFormat.Y16, 30),
            (640, 480, OBFormat.Y16, 15),
            (640, 400, OBFormat.Y16, 30),
            (640, 400, OBFormat.Y16, 15),
            (512, 512, OBFormat.Y16, 30),
            (512, 512, OBFormat.Y16, 15),
        ]

        depth_profile = _choose_depth_profile(profile_list, preferred, self.log)
        config.enable_stream(depth_profile)

        # Update spec to match what we actually opened (optional but useful)
        self.spec = DepthStreamSpec(
            width=depth_profile.get_width(),
            height=depth_profile.get_height(),
            fps=depth_profile.get_fps(),
            fmt=depth_profile.get_format(),
        )

        self._pipeline.start(config)
        self._started = True

        self.log.info(
            "Orbbec depth stream started: %dx%d @ %d FPS (%s)",
            self.spec.width,
            self.spec.height,
            self.spec.fps,
            self.spec.fmt,
        )

    def close(self) -> None:
        if self._pipeline and self._started:
            try:
                self._pipeline.stop()
            except Exception:
                self.log.exception("Failed to stop Orbbec pipeline")

        self._pipeline = None
        self._device = None
        self._ctx = None
        self._started = False

    def __enter__(self) -> "OrbbecDepthCamera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_depth(self) -> Optional[np.ndarray]:
        """
        Non-throwing grab.
        Returns None on timeout / missing frame.
        """
        MIN_DEPTH = 20  # 20mm
        MAX_DEPTH = 10000  # 10000mm

        if not self._started or self._pipeline is None:
            raise RuntimeError("Camera not opened. Call open() first.")

        frames = self._pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return None

        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            return None

        if depth_frame.get_format() != OBFormat.Y16:
            # You can relax this if you want to accept other formats, but Y16 is the normal depth path.
            self.log.debug("Unexpected depth format: %s", str(depth_frame.get_format()))
            return None

        w = int(depth_frame.get_width())
        h = int(depth_frame.get_height())

        # Depth scale is provided by SDK; docs multiply raw->scaled and cast.
        # We'll output millimeters as uint16 for downstream bg-sub/blob logic.
        scale_mm = float(depth_frame.get_depth_scale())  # millimeters per unit (typical)
        buf = depth_frame.get_data()

        depth_raw = np.frombuffer(buf, dtype=np.uint16).reshape((h, w))
        # Avoid "ndarray is not C-contiguous" style issues in some downstream ops
        depth_raw = np.ascontiguousarray(depth_raw)

        depth_mm = depth_raw.astype(np.float32) * scale_mm
        depth_mm = np.where((depth_raw > MIN_DEPTH) & (depth_raw < MAX_DEPTH), depth_raw, 0)
        depth_mm = depth_mm.astype(np.uint16)
        return depth_mm

    def get_size(self) -> Tuple[int, int]:
        return self.spec.width, self.spec.height

# ---------------------------
# Blob detection on foreground mask
# ---------------------------


@dataclasses.dataclass(frozen=True)
class Blob:
    label: int
    area: int
    centroid_xy: Tuple[float, float]  # (x,y) in image coordinates
    contour: np.ndarray               # Nx1x2 int32 for drawing


class ForegroundBlobDetector:
    """
    Foreground mask -> connected components -> filter by area -> contour extract -> centroids.
    """
    def __init__(self):
        pass

    @staticmethod
    def _find_contour_for_component(mask_component: np.ndarray) -> Optional[np.ndarray]:
        contours, _hier = cv2.findContours(mask_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # Take largest contour (defensive)
        return max(contours, key=cv2.contourArea)

    def detect(self, fg_mask_u8: np.ndarray, min_area: int, max_area: int) -> List[Blob]:
        """
        fg_mask_u8: uint8 0/255
        Returns list of Blob objects.
        """
        # connectedComponentsWithStats expects 0/255 or 0/1 both OK
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask_u8, connectivity=8)
        blobs: List[Blob] = []

        # label 0 is background
        for lab in range(1, num_labels):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue

            cx, cy = centroids[lab]
            # Build per-component mask for contour extraction (fast enough for modest blob counts)
            component_mask = (labels == lab).astype(np.uint8) * 255
            contour = self._find_contour_for_component(component_mask)
            if contour is None or contour.size == 0:
                continue

            blobs.append(
                Blob(
                    label=lab,
                    area=area,
                    centroid_xy=(float(cx), float(cy)),
                    contour=contour,
                )
            )
        return blobs


# ---------------------------
# Particle field simulation
# ---------------------------

@dataclasses.dataclass
class Sources:
    positions: np.ndarray   # (S,2) float32 in [0,1]x[0,1] space
    colors_bgr: np.ndarray  # (S,3) uint8

    @staticmethod
    def random(n: int, rng: np.random.Generator) -> "Sources":
        # Keep sources away from edges a bit
        positions = rng.uniform(0.12, 0.88, size=(n, 2)).astype(np.float32)
        colors = np.zeros((n, 3), dtype=np.uint8)
        for i in range(n):
            # pleasant-ish palette: sample in HSV then convert to BGR
            hue = int((i * 179 / max(1, n)) % 179)
            sat = 170
            val = 240
            hsv = np.uint8([[[hue, sat, val]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            colors[i] = bgr
        return Sources(positions=positions, colors_bgr=colors)


class ParticleField:
    """
    A square-bounded particle field with:
    - multiple sources that emit particles in pulses
    - repelling forces from blob centroids
    - collision binning along perimeter
    """

    def __init__(self, canvas_px: int = 820, border_px: int = 26, bins_per_side: int = 32):
        self.canvas_px = int(canvas_px)
        self.border_px = int(border_px)
        self.inner_min = self.border_px
        self.inner_max = self.canvas_px - self.border_px
        self.bins_per_side = int(bins_per_side)
        self.total_bins = self.bins_per_side * 4
        self.collision_bins = np.zeros((self.total_bins,), dtype=np.int32)

        self.rng = np.random.default_rng(12345)
        self.sources = Sources.random(5, self.rng)

        # Particles: store as struct-of-arrays for speed
        self.pos = np.zeros((0, 2), dtype=np.float32)   # px coords
        self.vel = np.zeros((0, 2), dtype=np.float32)
        self.col = np.zeros((0, 3), dtype=np.uint8)     # BGR
        self.age = np.zeros((0,), dtype=np.float32)     # seconds

    def reset_bins(self) -> None:
        self.collision_bins[:] = 0

    def set_num_sources(self, n: int) -> None:
        n = max(1, int(n))
        if n == self.sources.positions.shape[0]:
            return
        self.sources = Sources.random(n, self.rng)

    def _emit(self, particles_per_source: int, speed: float) -> None:
        S = self.sources.positions.shape[0]
        if particles_per_source <= 0:
            return

        # source positions in px
        src_px = self.inner_min + self.sources.positions * (self.inner_max - self.inner_min)
        src_px = src_px.astype(np.float32)

        # Emit N per source
        n_new = S * particles_per_source
        new_pos = np.repeat(src_px, repeats=particles_per_source, axis=0)

        # random directions
        angles = self.rng.uniform(0.0, 2.0 * math.pi, size=(n_new,)).astype(np.float32)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)

        # small random speed jitter
        sp = (speed * self.rng.uniform(0.7, 1.3, size=(n_new,)).astype(np.float32)).reshape(-1, 1)
        new_vel = dirs * sp

        # assign color by source
        new_col = np.repeat(self.sources.colors_bgr, repeats=particles_per_source, axis=0)

        new_age = np.zeros((n_new,), dtype=np.float32)

        self.pos = np.vstack([self.pos, new_pos])
        self.vel = np.vstack([self.vel, new_vel])
        self.col = np.vstack([self.col, new_col])
        self.age = np.concatenate([self.age, new_age])

    def _perimeter_bin_index(self, x: float, y: float) -> int:
        """
        Map a collision point on boundary to a perimeter bin.
        Bins go around clockwise: top (0..B-1), right, bottom, left.
        """
        B = self.bins_per_side
        # Clamp to inner bounds
        x = float(np.clip(x, self.inner_min, self.inner_max))
        y = float(np.clip(y, self.inner_min, self.inner_max))

        # Determine which side we hit (closest boundary)
        d_top = abs(y - self.inner_min)
        d_bot = abs(y - self.inner_max)
        d_left = abs(x - self.inner_min)
        d_right = abs(x - self.inner_max)
        m = min(d_top, d_right, d_bot, d_left)

        if m == d_top:
            t = (x - self.inner_min) / max(1e-6, (self.inner_max - self.inner_min))
            return int(np.clip(math.floor(t * B), 0, B - 1))
        if m == d_right:
            t = (y - self.inner_min) / max(1e-6, (self.inner_max - self.inner_min))
            return B + int(np.clip(math.floor(t * B), 0, B - 1))
        if m == d_bot:
            t = (self.inner_max - x) / max(1e-6, (self.inner_max - self.inner_min))
            return 2 * B + int(np.clip(math.floor(t * B), 0, B - 1))
        # left
        t = (self.inner_max - y) / max(1e-6, (self.inner_max - self.inner_min))
        return 3 * B + int(np.clip(math.floor(t * B), 0, B - 1))

    def _apply_repellers(self, repellers_px: np.ndarray, radius: float, strength: float) -> None:
        """
        repellers_px: (K,2) in px
        applies a repelling force to particles within radius.
        """
        if repellers_px.size == 0 or self.pos.shape[0] == 0:
            return

        r2 = float(radius * radius)
        for k in range(repellers_px.shape[0]):
            rp = repellers_px[k]
            d = self.pos - rp  # (N,2)
            dist2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
            m = dist2 < r2
            if not np.any(m):
                continue

            # Force magnitude falls off with distance; avoid singularity
            dist = np.sqrt(dist2[m] + 1e-6).reshape(-1, 1)
            dirn = d[m] / dist
            # Smooth falloff: (1 - (dist/radius))^2
            w = (1.0 - (dist / max(1e-6, radius)))
            w = np.clip(w, 0.0, 1.0)
            f = strength * (w * w)
            self.vel[m] += dirn * f

    def _handle_collisions(self) -> None:
        if self.pos.shape[0] == 0:
            return

        x = self.pos[:, 0]
        y = self.pos[:, 1]
        vx = self.vel[:, 0]
        vy = self.vel[:, 1]

        hit_left = x < self.inner_min
        hit_right = x > self.inner_max
        hit_top = y < self.inner_min
        hit_bottom = y > self.inner_max

        hit_any = hit_left | hit_right | hit_top | hit_bottom
        if not np.any(hit_any):
            return

        idxs = np.where(hit_any)[0]
        for i in idxs:
            # clamp
            self.pos[i, 0] = float(np.clip(self.pos[i, 0], self.inner_min, self.inner_max))
            self.pos[i, 1] = float(np.clip(self.pos[i, 1], self.inner_min, self.inner_max))

            # reflect velocity component(s) that caused exit
            if hit_left[i] or hit_right[i]:
                self.vel[i, 0] = -self.vel[i, 0] * 0.9
            if hit_top[i] or hit_bottom[i]:
                self.vel[i, 1] = -self.vel[i, 1] * 0.9

            # bin collision
            b = self._perimeter_bin_index(self.pos[i, 0], self.pos[i, 1])
            self.collision_bins[b] += 1

    def step(
        self,
        dt: float,
        repellers_px: np.ndarray,
        emit_enable: bool,
        particles_per_source: int,
        emit_speed: float,
        max_particles: int,
        damping: float,
        repeller_radius: float,
        repeller_strength: float,
    ) -> None:
        # emit
        if emit_enable:
            self._emit(particles_per_source=particles_per_source, speed=emit_speed)

        # cap particle count (drop oldest)
        if self.pos.shape[0] > max_particles:
            extra = self.pos.shape[0] - max_particles
            self.pos = self.pos[extra:]
            self.vel = self.vel[extra:]
            self.col = self.col[extra:]
            self.age = self.age[extra:]

        # forces
        self._apply_repellers(repellers_px=repellers_px, radius=repeller_radius, strength=repeller_strength)

        # integrate
        if self.pos.shape[0] > 0:
            self.pos += self.vel * float(dt)
            self.vel *= float(damping)
            self.age += float(dt)

        # collisions & binning
        self._handle_collisions()

    def render(self, title: str = "Spore Field") -> np.ndarray:
        """
        Returns a BGR image for display.
        """
        img = np.zeros((self.canvas_px, self.canvas_px, 3), dtype=np.uint8)

        # draw inner square boundary
        cv2.rectangle(
            img,
            (self.inner_min, self.inner_min),
            (self.inner_max, self.inner_max),
            (240, 240, 240),
            2,
            lineType=cv2.LINE_AA,
        )

        # draw perimeter bin “heat strip”
        B = self.bins_per_side
        counts = self.collision_bins.astype(np.float32)
        mx = float(np.max(counts)) if counts.size else 1.0
        mx = max(mx, 1.0)
        norm = np.clip(counts / mx, 0.0, 1.0)

        # render bins as short segments around the square
        thickness = 10
        for i in range(B):
            v = float(norm[i])
            if v > 0:
                x0 = int(self.inner_min + (self.inner_max - self.inner_min) * (i / B))
                x1 = int(self.inner_min + (self.inner_max - self.inner_min) * ((i + 1) / B))
                cv2.line(img, (x0, self.inner_min), (x1, self.inner_min), (int(60 + 195 * v),) * 3, thickness)

        for i in range(B):
            v = float(norm[B + i])
            if v > 0:
                y0 = int(self.inner_min + (self.inner_max - self.inner_min) * (i / B))
                y1 = int(self.inner_min + (self.inner_max - self.inner_min) * ((i + 1) / B))
                cv2.line(img, (self.inner_max, y0), (self.inner_max, y1), (int(60 + 195 * v),) * 3, thickness)

        for i in range(B):
            v = float(norm[2 * B + i])
            if v > 0:
                x0 = int(self.inner_max - (self.inner_max - self.inner_min) * (i / B))
                x1 = int(self.inner_max - (self.inner_max - self.inner_min) * ((i + 1) / B))
                cv2.line(img, (x0, self.inner_max), (x1, self.inner_max), (int(60 + 195 * v),) * 3, thickness)

        for i in range(B):
            v = float(norm[3 * B + i])
            if v > 0:
                y0 = int(self.inner_max - (self.inner_max - self.inner_min) * (i / B))
                y1 = int(self.inner_max - (self.inner_max - self.inner_min) * ((i + 1) / B))
                cv2.line(img, (self.inner_min, y0), (self.inner_min, y1), (int(60 + 195 * v),) * 3, thickness)

        # draw sources (subtle)
        src_px = self.inner_min + self.sources.positions * (self.inner_max - self.inner_min)
        for i in range(src_px.shape[0]):
            p = (int(src_px[i, 0]), int(src_px[i, 1]))
            col = tuple(int(x) for x in self.sources.colors_bgr[i])
            cv2.circle(img, p, 6, col, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, p, 8, (230, 230, 230), 1, lineType=cv2.LINE_AA)

        # draw particles
        if self.pos.shape[0] > 0:
            pts = self.pos.astype(np.int32)
            for i in range(pts.shape[0]):
                x, y = int(pts[i, 0]), int(pts[i, 1])
                if x < 0 or x >= self.canvas_px or y < 0 or y >= self.canvas_px:
                    continue
                c = tuple(int(v) for v in self.col[i])
                cv2.circle(img, (x, y), 1, c, -1, lineType=cv2.LINE_AA)

        # title overlay
        cv2.putText(
            img,
            title,
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "b: capture bg | c: clear bg | r: reset bins | q/esc: quit",
            (16, self.canvas_px - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return img


# ---------------------------
# Main App
# ---------------------------

class App:
    def __init__(self, device_index: int):
        self.cam = OrbbecDepthCamera(
            spec=DepthStreamSpec(width=640, height=400, fps=15),
            timeout_ms=100,
            device_index=0,
        )
        self.cam.open()
        self.detector = ForegroundBlobDetector()
        self.particles = ParticleField(canvas_px=860, border_px=30, bins_per_side=36)

        self.bg_depth: Optional[np.ndarray] = None
        self.last_t = time.time()

        # Windows
        self.depth_win = "Depth / BG Sub / Blobs"
        self.ui_win = "Controls"

        cv2.namedWindow(self.depth_win, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.ui_win, cv2.WINDOW_NORMAL)

        # Controls live on ui_win
        self.ui = TrackbarUI(self.ui_win)

        # Depth / BG subtraction controls
        self.ui.add("DepthDiffThresh_mm", init=60, maxv=600)   # threshold in mm
        self.ui.add("MorphOpen", init=1, maxv=10)              # kernel size ~ (2k+1)
        self.ui.add("MorphClose", init=2, maxv=10)
        self.ui.add("MinDepth_mm", init=200, maxv=4000)        # ignore too-near
        self.ui.add("MaxDepth_mm", init=3500, maxv=12000)      # ignore too-far

        # Blob controls
        self.ui.add("MinBlobArea", init=700, maxv=30000)
        self.ui.add("MaxBlobArea", init=20000, maxv=200000)

        # Particle controls
        self.ui.add("NumSources", init=5, maxv=20)
        self.ui.add("EmitEnable", init=1, maxv=1)
        self.ui.add("ParticlesPerSrc", init=45, maxv=250)
        self.ui.add("EmitSpeed", init=12, maxv=80)
        self.ui.add("MaxParticles", init=9000, maxv=40000)
        self.ui.add("Damping_x1000", init=985, maxv=999)        # 0.985
        self.ui.add("RepelRadius_px", init=90, maxv=260)
        self.ui.add("RepelStrength", init=6, maxv=30)

    def _depth_to_display(self, depth_u16: np.ndarray, fg_mask_u8: np.ndarray) -> np.ndarray:
        """
        Make a nice depth visualization, masked to foreground where possible.
        """
        depth = depth_u16.copy()

        # Mask background to 0 for visualization
        depth[fg_mask_u8 == 0] = 0

        # Normalize for display
        disp = np.zeros(depth.shape, dtype=np.uint8)
        nonzero = depth > 0
        if np.any(nonzero):
            d = depth[nonzero].astype(np.float32)
            lo = float(np.percentile(d, 2))
            hi = float(np.percentile(d, 98))
            hi = max(hi, lo + 1.0)
            scaled = (np.clip(depth.astype(np.float32), lo, hi) - lo) * (255.0 / (hi - lo))
            disp = scaled.astype(np.uint8)

        color = cv2.applyColorMap(disp, cv2.COLORMAP_TURBO)
        color[disp == 0] = (0, 0, 0)
        return color

    def _compute_foreground_mask(self, depth_u16: np.ndarray) -> np.ndarray:
        """
        Returns uint8 0/255 foreground mask.
        """
        min_depth = self.ui.get("MinDepth_mm")
        max_depth = self.ui.get("MaxDepth_mm")
        thresh = self.ui.get("DepthDiffThresh_mm")

        valid = (depth_u16 >= min_depth) & (depth_u16 <= max_depth) & (depth_u16 != 0)

        if self.bg_depth is None:
            # If no background captured yet, treat everything valid as foreground
            fg = valid
        else:
            # Background subtraction on depth
            # Note: absdiff on uint16 is safe if cast to int32 first
            diff = np.abs(depth_u16.astype(np.int32) - self.bg_depth.astype(np.int32)).astype(np.int32)
            fg = valid & (diff >= thresh)

        mask = (fg.astype(np.uint8) * 255)

        # Morphology to clean up
        open_k = self.ui.get("MorphOpen")
        close_k = self.ui.get("MorphClose")

        if open_k > 0:
            k = 2 * open_k + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if close_k > 0:
            k = 2 * close_k + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def _blob_centroids_to_particle_space(self, centroids_xy: np.ndarray, depth_shape: Tuple[int, int]) -> np.ndarray:
        """
        Map image-space centroids (x,y) to particle canvas pixel coordinates.
        """
        if centroids_xy.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        h, w = depth_shape
        # normalize 0..1
        nx = centroids_xy[:, 0] / max(1.0, (w - 1))
        ny = centroids_xy[:, 1] / max(1.0, (h - 1))

        # map into inner square
        inner_min = self.particles.inner_min
        inner_max = self.particles.inner_max
        px = inner_min + nx * (inner_max - inner_min)
        py = inner_min + ny * (inner_max - inner_min)
        return np.stack([px, py], axis=1).astype(np.float32)

    def run(self) -> None:
        logging.info("Controls: b=capture background, c=clear background, r=reset bins, q/esc=quit")

        while True:
            now = time.time()
            dt = now - self.last_t
            self.last_t = now
            dt = float(np.clip(dt, 1e-3, 0.05))  # guard huge dt when debugging

            depth_mm = self.cam.read_depth()
            if depth_mm is None:
                logging.warning("Failed to read depth frame")
                continue

            # UI-driven source count
            self.particles.set_num_sources(self.ui.get("NumSources"))

            # Foreground
            fg_mask = self._compute_foreground_mask(depth_mm)

            # Blobs
            min_area = self.ui.get("MinBlobArea")
            max_area = self.ui.get("MaxBlobArea")
            blobs = self.detector.detect(fg_mask, min_area=min_area, max_area=max_area)

            # Depth visualization with overlays
            depth_vis = self._depth_to_display(depth_mm, fg_mask)

            # draw blob perimeters + centroids
            for b in blobs:
                cv2.drawContours(depth_vis, [b.contour], -1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cx, cy = b.centroid_xy
                cv2.circle(depth_vis, (int(cx), int(cy)), 4, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(depth_vis, (int(cx), int(cy)), 6, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # Simple HUD
            hud = f"bg={'YES' if self.bg_depth is not None else 'NO'}  blobs={len(blobs)}"
            cv2.putText(depth_vis, hud, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (10, 10, 10), 3, cv2.LINE_AA)
            cv2.putText(depth_vis, hud, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 1, cv2.LINE_AA)

            cv2.imshow(self.depth_win, depth_vis)

            # Particle sim step
            centroids = np.array([b.centroid_xy for b in blobs], dtype=np.float32) if blobs else np.zeros((0, 2), dtype=np.float32)
            repellers_px = self._blob_centroids_to_particle_space(centroids, depth_shape=depth_mm.shape[:2])

            emit_enable = bool(self.ui.get("EmitEnable"))
            particles_per_src = self.ui.get("ParticlesPerSrc")
            emit_speed = float(self.ui.get("EmitSpeed"))
            max_particles = self.ui.get("MaxParticles")
            damping = float(self.ui.get("Damping_x1000")) / 1000.0
            repel_radius = float(self.ui.get("RepelRadius_px"))
            repel_strength = float(self.ui.get("RepelStrength"))

            self.particles.step(
                dt=dt,
                repellers_px=repellers_px,
                emit_enable=emit_enable,
                particles_per_source=particles_per_src,
                emit_speed=emit_speed,
                max_particles=max_particles,
                damping=damping,
                repeller_radius=repel_radius,
                repeller_strength=repel_strength,
            )

            spore_img = self.particles.render(title="Spore Field")
            # draw repellers for debugging aesthetics (subtle)
            for rp in repellers_px:
                cv2.circle(spore_img, (int(rp[0]), int(rp[1])), 10, (230, 230, 230), 1, lineType=cv2.LINE_AA)

            cv2.imshow("Spore Field", spore_img)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q
                break
            if key == ord('b'):
                self.bg_depth = depth_mm.copy()
                logging.info("Captured background frame.")
            if key == ord('c'):
                self.bg_depth = None
                logging.info("Cleared background.")
            if key == ord('r'):
                self.particles.reset_bins()
                logging.info("Reset perimeter bins.")

        self.shutdown()

    def shutdown(self) -> None:
        self.cam.close()
        cv2.destroyAllWindows()


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-index", type=int, default=0, help="CAP_OBSENSOR device index (default 0).")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    app = App(device_index=args.device_index)
    app.run()


if __name__ == "__main__":
    main()
