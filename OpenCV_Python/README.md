# Femto Bolt Depth Blobs & Spore Field Visualization

This application captures depth data from an **Orbbec Femto Bolt** depth camera and uses it to drive a real‑time interactive visualization. New objects appearing in the depth scene repel particles in a simulated “spore field,” while collisions against the boundary accumulate into a perimeter heat map.

---

## Features

- **Robust Orbbec SDK–based depth capture**
  - Uses `pyorbbecsdk` instead of OpenCV UVC for stability
  - Explicit depth profile selection with fallback
- **User‑triggered background capture**
  - Press `b` to record a depth background
  - Subsequent frames are background‑subtracted
- **Real‑time blob detection**
  - Adjustable depth thresholds and blob area filters
  - Centroid extraction for each detected object
- **Dual visualization windows**
  1. **Depth / BG Sub / Blobs**
     - Depth colormap
     - Optional background removal
     - Blob contours + centroids
  2. **Spore Field**
     - Particle emitters (“mushroom spores”)
     - Object centroids act as repelling forces
     - Boundary collision binning visualized as a heat strip
- **Live parameter tuning**
  - OpenCV trackbars for depth, blob, and particle parameters

---

## Requirements

### Hardware
- Orbbec **Femto Bolt**
- USB 3.x / USB‑C port (direct motherboard connection strongly recommended)

### Software
- Python 3.9+
- Windows or Linux (tested on both)
- Orbbec SDK v2

### Python dependencies
```bash
pip install numpy opencv-python opencv-contrib-python pyorbbecsdk
```

> ⚠️ If you encounter strange buffer or conversion issues, pin NumPy:
> ```bash
> pip install "numpy<2"
> ```

---

## Running the Application

```bash
python femto_bolt_depth_blobs_spores.py
```

Optional flags (if enabled in your version):
```bash
--device-index 0
--verbose
```

---

## Controls

### Keyboard
- **b** — Capture background frame
- **c** — Clear background
- **r** — Reset perimeter collision bins
- **q / ESC** — Quit

### Trackbars
- Depth thresholds (min/max, background diff)
- Blob area limits
- Particle system parameters:
  - Number of sources
  - Emission rate & speed
  - Repulsion radius & strength
  - Max particle count

---

## Windows Overview

### 1. Depth / BG Sub / Blobs
- Shows the depth image
- Foreground mask and blob contours
- Centroids used for interaction logic

### 2. Spore Field
- Square boundary with particle emitters
- Particles repel away from detected objects
- Boundary collisions increment per‑edge bins

---

## Project Structure (Single‑File App)

- `OrbbecDepthCamera` — SDK‑based depth capture
- `ForegroundBlobDetector` — background subtraction + blob extraction
- `ParticleField` — particle simulation and collision binning
- `App` — UI, visualization, and control loop

The code is intentionally written in a **clear, inspectable style** to make debugging and extension easy.

---


## License / Use

This code is intended as a **prototype / research / installation tool**.  
Adapt, extend, and integrate freely.

