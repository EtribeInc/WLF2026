#!/usr/bin/env python3
"""
osc_validate_pipeline.py

Validate OSC messages published by:
  - feet_xy_osc_femto_bolt_validated*.py (per-camera /wlf/blobs + /wlf/frame)
  - osc_blob_fuser.py (fused output, e.g. /wlf/fused)

This script can load config from either:
  - YAML (the previous workflow), OR
  - the PowerShell launcher script run_wlf_osc_pipeline.ps1 (your current workflow).

Examples (PowerShell config):
  python osc_validate_from_yaml.py --config src/run_wlf_osc_pipeline.ps1 --mode cameras
  python osc_validate_from_yaml.py --config src/run_wlf_osc_pipeline.ps1 --mode fused
  python osc_validate_from_yaml.py --config src/run_wlf_osc_pipeline.ps1 --mode all

Examples (YAML config):
  python osc_validate_from_yaml.py --config src/wlf_osc_cameras.yaml --mode cameras
  python osc_validate_from_yaml.py --config src/wlf_osc_cameras.yaml --mode fused

Requirements:
  pip install python-osc
  For YAML configs only: pip install pyyaml
"""

import argparse
import re
import time
import threading
from collections import defaultdict

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer


# ----------------------------
# Config loaders
# ----------------------------

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def load_config_from_ps1(ps1_path: str) -> dict:
    """
    Extracts values from the USER CONFIG block of run_wlf_osc_pipeline.ps1-like scripts.
    This is intentionally minimal: it only reads the variables we need for validation.
    """
    with open(ps1_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Pull assignments like: $CAM0_PORT = 9001
    # We'll scan the whole file; later values win.
    assigns = {}
    for m in re.finditer(r'^\s*\$([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$', text, flags=re.MULTILINE):
        key = m.group(1)
        val = m.group(2).strip()

        # remove trailing comments
        val = re.sub(r'\s+#.*$', '', val).strip()

        # Normalize common PS literals
        if val.lower() in ("$true", "true"):
            assigns[key] = True
        elif val.lower() in ("$false", "false"):
            assigns[key] = False
        elif re.fullmatch(r'-?\d+', val):
            assigns[key] = int(val)
        elif re.fullmatch(r'-?\d+\.\d*', val):
            assigns[key] = float(val)
        else:
            assigns[key] = _strip_quotes(val)

    # Required keys (based on the launcher you pasted)
    cam_ports = []
    cam_names = []

    # Prefer CAM0/CAM1, but allow CAM2+ if you extend later.
    for i in range(0, 16):
        k = f"CAM{i}_PORT"
        if k in assigns:
            cam_ports.append(int(assigns[k]))
            cam_names.append(f"cam{i}")
    if not cam_ports:
        raise RuntimeError("No $CAM*_PORT variables found in PS1 config.")

    cfg = {
        "type": "ps1",
        "cameras": [{"name": n, "port": p} for n, p in zip(cam_names, cam_ports)],
        "osc": {
            "in_addr_blobs": str(assigns.get("IN_ADDR_BLOBS", "/wlf/blobs")),
            "in_addr_frame": str(assigns.get("IN_ADDR_FRAME", "/wlf/frame")),
            "fused_out_port": int(assigns.get("FUSED_OUT_PORT", 9100)),
            "fused_out_addr": str(assigns.get("FUSED_OUT_ADDR", "/wlf/fused")),
        },
        "blob_limits": {
            "max_blobs_per_cam": int(assigns.get("MAX_BLOBS_PER_CAM", 8)),
            "max_total_blobs": int(assigns.get("MAX_TOTAL_BLOBS", len(cam_ports) * int(assigns.get("MAX_BLOBS_PER_CAM", 8)))),
        },
    }
    return cfg


def load_config_from_yaml(yaml_path: str) -> dict:
    import yaml  # optional dependency
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    # Support both old and new YAML shapes
    if "cameras" in y and "osc" in y and "blob_limits" in y:
        cams = [{"name": c["name"], "port": int(c["in_port"])} for c in y["cameras"]]
        return {
            "type": "yaml",
            "cameras": cams,
            "osc": {
                "in_addr_blobs": y["osc"]["in_addr_blobs"],
                "in_addr_frame": y["osc"].get("in_addr_frame", "/wlf/frame"),
                "fused_out_port": int(y["osc"].get("fused_out_port", y.get("fuser", {}).get("out_port", 9100))),
                "fused_out_addr": y["osc"].get("fused_out_addr", y.get("fuser", {}).get("out_addr", "/wlf/fused")),
            },
            "blob_limits": {
                "max_blobs_per_cam": int(y["blob_limits"]["max_blobs_per_cam"]),
                "max_total_blobs": int(y["blob_limits"]["max_total_blobs"]),
            },
        }

    # Older YAML variant (if ever used)
    raise RuntimeError("Unrecognized YAML schema (expected cameras/osc/blob_limits).")


def load_config(path: str) -> dict:
    if path.lower().endswith(".ps1"):
        return load_config_from_ps1(path)
    if path.lower().endswith((".yaml", ".yml")):
        return load_config_from_yaml(path)
    raise RuntimeError("Config file must be .ps1, .yaml, or .yml")


# ----------------------------
# OSC validators
# ----------------------------

def make_validator(addr_blobs: str, addr_frame: str, max_blobs: int):
    stats = defaultdict(int)
    last_seen = {}

    def handler(address, *args):
        now = time.time()
        stats["msgs_total"] += 1
        last_seen[address] = now

        if address == addr_frame:
            # Expected: [frame_idx(int), n(int)]
            if len(args) != 2:
                stats["frame_bad_len"] += 1
                stats["frame_bad_len_last"] = len(args)
                return
            try:
                int(args[0]); int(args[1])
            except Exception:
                stats["frame_bad_types"] += 1
                return
            stats["frame_ok"] += 1
            return

        if address == addr_blobs:
            expected_len = 2 + 2 * int(max_blobs)
            if len(args) != expected_len:
                stats["blobs_bad_len"] += 1
                stats["blobs_bad_len_last"] = len(args)
                return

            try:
                frame_idx = int(args[0])
                n = int(args[1])
            except Exception:
                stats["blobs_bad_header_types"] += 1
                return

            if n < 0 or n > int(max_blobs):
                stats["blobs_bad_n"] += 1
                return

            for i in range(2, expected_len, 2):
                try:
                    float(args[i]); float(args[i + 1])
                except Exception:
                    stats["blobs_bad_xy_types"] += 1
                    return

            stats["blobs_ok"] += 1
            stats["last_frame"] = frame_idx
            stats["last_n"] = n
            return

        stats["msgs_other_addr"] += 1

    return handler, stats, last_seen


def make_fused_validator(addr_fused: str, max_total_blobs: int):
    stats = defaultdict(int)
    last_seen = {}

    def handler(address, *args):
        now = time.time()
        stats["msgs_total"] += 1
        last_seen[address] = now

        if address != addr_fused:
            stats["msgs_other_addr"] += 1
            return

        expected_len = 2 + 2 * int(max_total_blobs)
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

        if n < 0 or n > int(max_total_blobs):
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


def serve(port: int, dispatcher: Dispatcher):
    server = ThreadingOSCUDPServer(("0.0.0.0", int(port)), dispatcher)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to run_wlf_osc_pipeline.ps1 OR wlf_osc_cameras.yaml")
    ap.add_argument("--mode", choices=["cameras", "fused", "all"], default="cameras")
    ap.add_argument("--print-every", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    osc = cfg["osc"]
    limits = cfg["blob_limits"]

    servers = []
    t0 = time.time()
    last_print = 0.0

    if args.mode in ("cameras", "all"):
        addr_blobs = osc["in_addr_blobs"]
        addr_frame = osc["in_addr_frame"]
        max_blobs = int(limits["max_blobs_per_cam"])

        for cam in cfg["cameras"]:
            disp = Dispatcher()
            handler, stats, last_seen = make_validator(addr_blobs, addr_frame, max_blobs)
            disp.set_default_handler(handler)
            server = serve(int(cam["port"]), disp)
            servers.append(("cam:" + cam["name"], server, stats, last_seen, {"blobs": addr_blobs, "frame": addr_frame, "fused": None}))
            print(f"Listening {cam['name']} UDP {cam['port']} validating blobs={addr_blobs} frame={addr_frame} max_blobs={max_blobs}")

    if args.mode in ("fused", "all"):
        addr_fused = osc["fused_out_addr"]
        max_total = int(limits["max_total_blobs"])
        disp = Dispatcher()
        handler, stats, last_seen = make_fused_validator(addr_fused, max_total)
        disp.set_default_handler(handler)
        server = serve(int(osc["fused_out_port"]), disp)
        servers.append(("fused", server, stats, last_seen, {"blobs": None, "frame": None, "fused": addr_fused}))
        print(f"Listening fused UDP {osc['fused_out_port']} validating addr={addr_fused} max_blobs={max_total}")

    if not servers:
        raise RuntimeError("No servers started (check --mode).")

    try:
        while True:
            time.sleep(0.05)
            now = time.time()
            if now - last_print >= args.print_every:
                last_print = now
                dt = now - t0
                print("\n--- OSC Validation Stats (t=%.1fs) ---" % dt)
                for name, server, stats, last_seen, addrs in servers:
                    port = server.server_address[1]
                    total = stats.get("msgs_total", 0)
                    other = stats.get("msgs_other_addr", 0)

                    if name.startswith("cam:"):
                        blobs_ok = stats.get("blobs_ok", 0)
                        frame_ok = stats.get("frame_ok", 0)
                        blobs_bad_len = stats.get("blobs_bad_len", 0)
                        frame_bad_len = stats.get("frame_bad_len", 0)
                        blobs_bad_n = stats.get("blobs_bad_n", 0)
                        last_frame = stats.get("last_frame", None)
                        last_n = stats.get("last_n", None)

                        last_blobs = last_seen.get(addrs["blobs"], None)
                        last_frame_seen = last_seen.get(addrs["frame"], None)
                        age_blobs = (now - last_blobs) if last_blobs else None
                        age_frame = (now - last_frame_seen) if last_frame_seen else None
                        ageb = ("%.2fs" % age_blobs) if age_blobs is not None else "never"
                        agef = ("%.2fs" % age_frame) if age_frame is not None else "never"

                        print(
                            f"{name:>8} port={port} total={total} other={other} "
                            f"blobs_ok={blobs_ok} blobs_bad_len={blobs_bad_len} blobs_bad_n={blobs_bad_n} "
                            f"frame_ok={frame_ok} frame_bad_len={frame_bad_len} "
                            f"last_frame={last_frame} last_n={last_n} last_blobs={ageb} last_frame_msg={agef}"
                        )
                    else:
                        ok = stats.get("ok", 0)
                        bad_len = stats.get("bad_len", 0)
                        bad_n = stats.get("bad_n", 0)
                        last_frame = stats.get("last_frame", None)
                        last_n = stats.get("last_n", None)
                        last = last_seen.get(addrs["fused"], None)
                        age = (now - last) if last else None
                        ages = ("%.2fs" % age) if age is not None else "never"

                        print(
                            f"{name:>8} port={port} total={total} other={other} ok={ok} bad_len={bad_len} bad_n={bad_n} "
                            f"last_frame={last_frame} last_n={last_n} last_seen={ages}"
                        )

    except KeyboardInterrupt:
        print("\nStopping.")


if __name__ == "__main__":
    main()
