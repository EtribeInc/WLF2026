#!/usr/bin/env python3
"""
osc_blob_fuser.py

Listens to fixed-size OSC blob streams from multiple camera scripts, applies per-camera
XY offsets (and optional yaw rotation), normalizes into a shared [0,1]x[0,1] space,
and publishes a fused fixed-size OSC payload for TouchDesigner.

Input OSC contract (from feet_xy_osc_femto_bolt.py)
---------------------------------------------------
/wlf/blobs:
  [frame_idx(int), num_blobs(int),
   x0, y0, x1, y1, ... x(N-1), y(N-1)]
Where N = max_blobs (fixed) and unused pairs are padded (often NaN).

Note: We do NOT need an explicit camera ID in the OSC payload as long as each camera
stream arrives on a distinct UDP port (recommended). Each port corresponds to one camera.

Fusion
------
For each camera i, each point p_cam (meters) is transformed into shared world meters:
  p_world = R(yaw_i) * p_cam + t_i
where t_i = [tx, ty] is the known installed offset and yaw_i is optional.

Then points are concatenated (truncated to max_total_blobs) and normalized:
  u = (x - minx) / (maxx - minx)
  v = (y - miny) / (maxy - miny)
clamped to [0,1].

Output OSC contract (fixed size)
--------------------------------
/wlf/fused:
  [frame_seq(int), num_pts(int),
   u0, v0, u1, v1, ... u(M-1), v(M-1)]
Where M = max_total_blobs (fixed), padded with pad_value (default NaN).

Dependencies
------------
  pip install python-osc numpy

Example
-------
Assuming 3 camera scripts publishing to ports 9001,9002,9003:

python osc_blob_fuser.py \
  --in cam1:9001 cam2:9002 cam3:9003 \
  --offset cam1:0:0 cam2:2.0:0 cam3:0:2.0 \
  --yaw cam3:90 \
  --bounds -3 3 -3 3 \
  --out-host 127.0.0.1 --out-port 9100 \
  --max-blobs-per-cam 8 --max-total-blobs 24
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient


@dataclass
class CamConfig:
    name: str
    listen_port: int
    tx: float
    ty: float
    yaw_deg: float


@dataclass
class LatestFrame:
    t_recv: float
    frame_idx: int
    num_blobs: int
    pts_xy: List[Tuple[float, float]]


def rot2d(yaw_deg: float) -> np.ndarray:
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def parse_kv_list(items: List[str]) -> Dict[str, List[str]]:
    """
    Parses ["cam1:9001", "cam2:9002"] into {"cam1":["9001"], "cam2":["9002"]}.
    Also supports name:val:val...
    """
    out: Dict[str, List[str]] = {}
    for it in items:
        parts = it.split(":")
        if len(parts) < 2:
            raise ValueError(f"Bad item '{it}'. Expected name:val[:val...]")
        name = parts[0]
        out[name] = parts[1:]
    return out


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class Fuser:
    def __init__(
        self,
        cams: List[CamConfig],
        max_blobs_per_cam: int,
        max_total_blobs: int,
        bounds: Tuple[float, float, float, float],  # minx maxx miny maxy
        out_client: SimpleUDPClient,
        out_addr: str,
        pad_value: float,
        stale_s: float,
        publish_hz: float,
        in_addr_blobs: str,
    ):
        self.cams = {c.name: c for c in cams}
        self.max_blobs_per_cam = max_blobs_per_cam
        self.max_total_blobs = max_total_blobs
        self.minx, self.maxx, self.miny, self.maxy = bounds
        self.out = out_client
        self.out_addr = out_addr
        self.pad_value = pad_value
        self.stale_s = stale_s
        self.publish_dt = 1.0 / publish_hz if publish_hz > 0 else 0.0
        self.in_addr_blobs = in_addr_blobs

        self.latest: Dict[str, LatestFrame] = {}
        self.frame_seq = 0

        # precompute transforms
        self.R: Dict[str, np.ndarray] = {}
        self.t: Dict[str, np.ndarray] = {}
        for name, c in self.cams.items():
            self.R[name] = rot2d(c.yaw_deg)
            self.t[name] = np.array([c.tx, c.ty], dtype=np.float32)

    def on_blobs(self, cam_name: str, address: str, *args):
        # args: [frame_idx(int), num_blobs(int), x0, y0, ...]
        if len(args) < 2:
            return
        try:
            frame_idx = int(args[0])
            num_blobs = int(args[1])
        except Exception:
            return

        vals = args[2:]
        pts: List[Tuple[float, float]] = []

        k = min(num_blobs, self.max_blobs_per_cam)
        for i in range(k):
            j = 2 * i
            if j + 1 >= len(vals):
                break
            x = float(vals[j])
            y = float(vals[j + 1])
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            pts.append((x, y))

        self.latest[cam_name] = LatestFrame(
            t_recv=time.time(),
            frame_idx=frame_idx,
            num_blobs=len(pts),
            pts_xy=pts,
        )

    def fuse_once(self):
        now = time.time()
        fused: List[Tuple[float, float]] = []

        for cam_name, lf in list(self.latest.items()):
            if (now - lf.t_recv) > self.stale_s:
                continue
            if cam_name not in self.cams:
                continue

            R = self.R[cam_name]
            t = self.t[cam_name]
            for (x, y) in lf.pts_xy:
                p = np.array([x, y], dtype=np.float32)
                pw = (R @ p) + t
                fused.append((float(pw[0]), float(pw[1])))

        fused = fused[: self.max_total_blobs]

        sx = (self.maxx - self.minx)
        sy = (self.maxy - self.miny)
        normed: List[Tuple[float, float]] = []
        if sx > 1e-9 and sy > 1e-9:
            for x, y in fused:
                u = (x - self.minx) / sx
                v = (y - self.miny) / sy
                normed.append((clamp01(u), clamp01(v)))

        n = len(normed)
        payload: List[float] = [int(self.frame_seq), int(n)]
        for (u, v) in normed:
            payload.append(float(u))
            payload.append(float(v))
        for _ in range(self.max_total_blobs - n):
            payload.append(float(self.pad_value))
            payload.append(float(self.pad_value))

        self.out.send_message(self.out_addr, payload)
        self.frame_seq += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="Input listeners as name:port. Example: cam1:9001 cam2:9002")
    ap.add_argument("--offset", nargs="+", required=True,
                    help="Per-camera offsets as name:tx:ty (meters). Example: cam1:0:0 cam2:2.0:0")
    ap.add_argument("--yaw", nargs="*", default=[],
                    help="Optional yaw as name:deg. Example: cam2:90")

    ap.add_argument("--in-addr-blobs", type=str, default="/wlf/blobs",
                    help="OSC address to listen for incoming blob lists.")
    ap.add_argument("--bounds", nargs=4, type=float, required=True,
                    metavar=("MINX", "MAXX", "MINY", "MAXY"),
                    help="World bounds in meters for normalization to 0..1.")
    ap.add_argument("--max-blobs-per-cam", type=int, default=8)
    ap.add_argument("--max-total-blobs", type=int, default=24)
    ap.add_argument("--pad-value", type=float, default=float("nan"))
    ap.add_argument("--stale-s", type=float, default=0.5, help="Drop camera data older than this.")
    ap.add_argument("--publish-hz", type=float, default=30.0, help="Fused output rate.")

    ap.add_argument("--out-host", type=str, default="127.0.0.1")
    ap.add_argument("--out-port", type=int, default=9100)
    ap.add_argument("--out-addr", type=str, default="/wlf/fused")

    args = ap.parse_args()

    ins = parse_kv_list(args.inputs)
    offs = parse_kv_list(args.offset)
    yaws = parse_kv_list(args.yaw) if args.yaw else {}

    cams: List[CamConfig] = []
    for name, port_s in ins.items():
        if name not in offs:
            raise SystemExit(f"Missing --offset for camera '{name}'")
        if len(port_s) != 1:
            raise SystemExit(f"Bad --in for '{name}', expected name:port")

        if len(offs[name]) != 2:
            raise SystemExit(f"Bad --offset for '{name}', expected name:tx:ty")

        port = int(port_s[0])
        tx = float(offs[name][0])
        ty = float(offs[name][1])
        yaw = float(yaws[name][0]) if (name in yaws and len(yaws[name]) >= 1) else 0.0

        cams.append(CamConfig(name=name, listen_port=port, tx=tx, ty=ty, yaw_deg=yaw))

    out_client = SimpleUDPClient(args.out_host, args.out_port)
    fuser = Fuser(
        cams=cams,
        max_blobs_per_cam=args.max_blobs_per_cam,
        max_total_blobs=args.max_total_blobs,
        bounds=(args.bounds[0], args.bounds[1], args.bounds[2], args.bounds[3]),
        out_client=out_client,
        out_addr=args.out_addr,
        pad_value=args.pad_value,
        stale_s=args.stale_s,
        publish_hz=args.publish_hz,
        in_addr_blobs=args.in_addr_blobs,
    )

    # One OSC server per input port (in threads) to keep it simple.
    import threading

    def serve(cam_name: str, port: int):
        disp = Dispatcher()
        disp.map(args.in_addr_blobs, lambda addr, *a: fuser.on_blobs(cam_name, addr, *a))
        server = BlockingOSCUDPServer(("0.0.0.0", port), disp)
        server.serve_forever()

    for c in cams:
        t = threading.Thread(target=serve, args=(c.name, c.listen_port), daemon=True)
        t.start()

    # Publish loop
    try:
        while True:
            t0 = time.time()
            fuser.fuse_once()
            if fuser.publish_dt > 0:
                dt = time.time() - t0
                sleep_s = fuser.publish_dt - dt
                if sleep_s > 0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
