#!/usr/bin/env python3
"""
osc_validate_pipeline.py

Consume the same YAML as the launcher (authoritative) and validate OSC messages:

Modes:
  --mode cameras  : listen to each camera in_port and validate /wlf/blobs payloads
  --mode fused    : listen to fused_out_port and validate fused_out_addr payload

Examples:
  python osc_validate_pipeline.py --yaml src/wlf_osc_cameras.yaml --mode cameras
  python osc_validate_pipeline.py --yaml src/wlf_osc_cameras.yaml --mode fused

Requirements:
  pip install python-osc pyyaml
"""

import argparse
import time
import threading
from collections import defaultdict

import yaml
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer


def make_validator(addr_expected: str, max_blobs: int):
    stats = defaultdict(int)
    last_seen = {}

    def handler(address, *args):
        now = time.time()
        stats["msgs_total"] += 1
        last_seen[address] = now

        if address != addr_expected:
            stats["msgs_other_addr"] += 1
            return

        # /wlf/frame -> [frame_idx, n]
        if address.endswith("/frame"):
            if len(args) != 2:
                stats["bad_len"] += 1
                stats["bad_len_last"] = len(args)
                return
            stats["ok"] += 1
            return

        expected_len = 2 + 2 * int(max_blobs)
        if len(args) != expected_len:
            stats["bad_len"] += 1
            stats["bad_len_last"] = len(args)
            return

        try:
            frame_idx = int(args[0])
            n = int(args[1])
        except Exception:
            stats["bad_header_types"] += 1
            return

        if n < 0 or n > int(max_blobs):
            stats["bad_n"] += 1
            return

        for i in range(2, expected_len, 2):
            try:
                float(args[i]); float(args[i + 1])
            except Exception:
                stats["bad_xy_types"] += 1
                return

        stats["ok"] += 1
        stats["last_frame"] = frame_idx
        stats["last_n"] = n

    return handler, stats, last_seen


def serve(port: int, addr_expected: str, max_blobs: int):
    disp = Dispatcher()
    handler, stats, last_seen = make_validator(addr_expected, max_blobs)
    disp.set_default_handler(handler)

    server = ThreadingOSCUDPServer(("0.0.0.0", int(port)), disp)

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"Listening UDP {port} validating addr={addr_expected} max_blobs={max_blobs}")

    return server, stats, last_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="Path to wlf_osc_cameras.yaml")
    ap.add_argument("--mode", choices=["cameras", "fused"], default="cameras")
    ap.add_argument("--print-every", type=float, default=1.0)
    args = ap.parse_args()

    with open(args.yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    osc = cfg["osc"]
    limits = cfg["blob_limits"]

    servers = []
    t0 = time.time()
    last_print = 0.0

    if args.mode == "cameras":
        addr = osc["in_addr_blobs"]
        max_blobs = int(limits["max_blobs_per_cam"])
        for cam in cfg["cameras"]:
            servers.append((cam["name"],) + serve(int(cam["in_port"]), addr, max_blobs))
    else:
        addr = osc["fused_out_addr"]
        max_blobs = int(limits["max_total_blobs"])
        servers.append(("fused",) + serve(int(osc["fused_out_port"]), addr, max_blobs))

    try:
        while True:
            time.sleep(0.05)
            now = time.time()
            if now - last_print >= args.print_every:
                last_print = now
                dt = now - t0
                print("\n--- OSC Validation Stats (t=%.1fs) ---" % dt)
                for name, server, stats, last_seen in servers:
                    ok = stats.get("ok", 0)
                    total = stats.get("msgs_total", 0)
                    bad_len = stats.get("bad_len", 0)
                    bad_n = stats.get("bad_n", 0)
                    other = stats.get("msgs_other_addr", 0)
                    last_frame = stats.get("last_frame", None)
                    last_n = stats.get("last_n", None)
                    last = last_seen.get(addr, None)
                    age = (now - last) if last else None
                    age_s = ("%.2fs" % age) if age is not None else "never"
                    last_bad_len = stats.get("bad_len_last", None)
                    lb = f" last_bad_len={last_bad_len}" if last_bad_len is not None else ""
                    print(f"{name:>5} port={server.server_address[1]} total={total} ok={ok} bad_len={bad_len} bad_n={bad_n} other_addr={other} last_frame={last_frame} last_n={last_n} last_seen={age_s}{lb}")
    except KeyboardInterrupt:
        print("\nStopping.")


if __name__ == "__main__":
    main()
