#!/usr/bin/env python3
"""
osc_fused_test_publisher.py

Standalone OSC test publisher that emits representative /wlf/fused packets for TouchDesigner (or any OSC client).

Packet format (fixed-length, matches osc_blob_fuser.py output shape):
  /wlf/fused  [frame_idx(int), n(int), x0(float), y0(float), x1, y1, ...]  (total args = 2 + 2*max_total_blobs)

- Coordinates are normalized to (0..1) by default.
- The publisher generates a few "active" blobs (n) that bounce around the space.
- Remaining blob slots are filled with a configurable sentinel (default: -1.0) so the payload stays fixed-length.

Install:
  pip install python-osc

Examples:
  # Send to local TD listening on UDP 9100
  python osc_fused_test_publisher.py --host 127.0.0.1 --port 9100 --addr /wlf/fused

  # Faster rate, 2 active blobs, different seed
  python osc_fused_test_publisher.py --hz 60 --n 2 --seed 42

  # Use NaN fill for unused slots (some clients dislike NaN; default is -1)
  python osc_fused_test_publisher.py --fill nan
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

from pythonosc.udp_client import SimpleUDPClient


@dataclass
class Blob:
    x: float
    y: float
    vx: float
    vy: float


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def step_bounce(b: Blob, dt: float, speed_jitter: float = 0.0) -> None:
    """
    Integrate blob position with simple wall-bounce in [0,1]x[0,1].
    Optionally add tiny random speed jitter to avoid perfectly periodic motion.
    """
    if speed_jitter > 0.0:
        b.vx += random.uniform(-speed_jitter, speed_jitter) * dt
        b.vy += random.uniform(-speed_jitter, speed_jitter) * dt

    b.x += b.vx * dt
    b.y += b.vy * dt

    # Bounce off walls; reflect velocity and clamp position into [0,1]
    if b.x < 0.0:
        b.x = -b.x
        b.vx = abs(b.vx)
    elif b.x > 1.0:
        b.x = 2.0 - b.x
        b.vx = -abs(b.vx)

    if b.y < 0.0:
        b.y = -b.y
        b.vy = abs(b.vy)
    elif b.y > 1.0:
        b.y = 2.0 - b.y
        b.vy = -abs(b.vy)

    b.x = clamp01(b.x)
    b.y = clamp01(b.y)


def make_blobs(n: int, base_speed: float) -> List[Blob]:
    blobs: List[Blob] = []
    for _ in range(n):
        x = random.random()
        y = random.random()
        # random direction
        ang = random.random() * 2.0 * math.pi
        sp = base_speed * (0.7 + 0.6 * random.random())
        vx = math.cos(ang) * sp
        vy = math.sin(ang) * sp
        blobs.append(Blob(x, y, vx, vy))
    return blobs


def fill_value(fill: str) -> float:
    fill = fill.lower()
    if fill in ("-1", "-1.0", "neg1", "minus1"):
        return -1.0
    if fill in ("0", "0.0", "zero"):
        return 0.0
    if fill in ("nan",):
        return float("nan")
    raise ValueError("fill must be one of: -1, 0, nan")


def build_payload(frame_idx: int, blobs: List[Blob], max_total: int, fill: float) -> List[float]:
    """
    Build fixed-length payload: [frame_idx, n, x0, y0, ... x(max_total-1), y(max_total-1)]
    """
    n = min(len(blobs), max_total)
    payload: List[float] = [int(frame_idx), int(n)]

    # active blobs
    for i in range(n):
        payload.append(float(blobs[i].x))
        payload.append(float(blobs[i].y))

    # inactive slots
    for _ in range(max_total - n):
        payload.append(float(fill))
        payload.append(float(fill))

    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="Destination host (TD machine IP)")
    ap.add_argument("--port", type=int, default=9100, help="Destination UDP port TD listens on")
    ap.add_argument("--addr", default="/wlf/fused", help="OSC address pattern")
    ap.add_argument("--hz", type=float, default=30.0, help="Publish rate (Hz)")
    ap.add_argument("--max-total-blobs", type=int, default=16, help="Fixed max blob slots in payload")
    ap.add_argument("--n", type=int, default=3, help="Number of active blobs (<= max-total-blobs)")
    ap.add_argument("--speed", type=float, default=0.35, help="Base speed in normalized units/sec")
    ap.add_argument("--speed-jitter", type=float, default=0.0, help="Small random accel to avoid perfect periodicity (units/sec^2)")
    ap.add_argument("--fill", default="-1", help="Fill value for inactive slots: -1, 0, nan")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (int) for repeatable motion")
    ap.add_argument("--print-every", type=float, default=2.0, help="Print status every N seconds (0 disables)")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.n < 0 or args.n > args.max_total_blobs:
        raise SystemExit("--n must be in [0, max-total-blobs]")

    dt_target = 1.0 / max(args.hz, 0.1)
    fill = fill_value(args.fill)

    client = SimpleUDPClient(args.host, int(args.port))
    blobs = make_blobs(args.n, args.speed)

    print(f"Publishing OSC to {args.host}:{args.port} addr={args.addr}")
    print(f"rate={args.hz}Hz max_total_blobs={args.max_total_blobs} active_n={args.n} fill={args.fill}")
    print("Ctrl+C to stop.\n")

    frame_idx = 0
    t_prev = time.perf_counter()
    t_print = 0.0
    sent = 0

    try:
        while True:
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now

            # integrate blobs
            for b in blobs:
                step_bounce(b, dt, speed_jitter=args.speed_jitter)

            payload = build_payload(frame_idx, blobs, args.max_total_blobs, fill)

            # Send as OSC args. python-osc will encode ints/floats appropriately.
            client.send_message(args.addr, payload)

            sent += 1
            frame_idx += 1

            # Optional status
            if args.print_every and args.print_every > 0:
                t_print += dt
                if t_print >= args.print_every:
                    t_print = 0.0
                    # print first blob(s) for sanity
                    head = [(blobs[i].x, blobs[i].y) for i in range(min(len(blobs), 3))]
                    print(f"sent={sent} last_frame={frame_idx-1} n={args.n} head={[(round(x,3), round(y,3)) for x,y in head]}")

            # Sleep to maintain target rate
            t_after = time.perf_counter()
            work = t_after - t_now
            to_sleep = dt_target - work
            if to_sleep > 0:
                time.sleep(to_sleep)
    except KeyboardInterrupt:
        print("\nStopping.")


if __name__ == "__main__":
    main()
