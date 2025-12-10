# rkbx_wave Architecture

A cheat-sheet for understanding the signal chain, data flow, and debugging.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              rkbx_wave.py (GUI)                             │
│  WaveformSyncApp: Tkinter GUI with dual-deck waveform display               │
│  ├── GUI Controls → color_cfg, render_cfg                                   │
│  ├── OSC Listener → deck_idx, track_path, current_time, bpm                 │
│  └── Canvas displays ← PIL Images                                           │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │ calls
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         deck_controller.py                                  │
│  DeckController: Orchestrates single deck - ANLZ loading, BPM, rendering    │
│  ├── load_anlz(folder) → WaveformAnalysis, beat_grid                        │
│  ├── update_time(t) / update_live_bpm(bpm)                                  │
│  └── render() → RenderResult (PIL Image + timing info)                      │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │ calls
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            playhead.py                                      │
│  Window planning and timing calculations                                    │
│  ├── compute_timing_info() → TimingInfo (duration, bins, seconds_per_bin)   │
│  ├── compute_window_plan() → WindowPlan (start/end bins, playhead fraction) │
│  ├── ensure_prerender_cache() → PrerenderCache (full-track image)           │
│  ├── render_window_image() → crops/renders visible window                   │
│  └── finalize_preview_image() → scaled Image + playhead line                │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │ calls
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             render.py                                       │
│  PIL image generation from WaveformAnalysis                                 │
│  ├── render_waveform_image() → draws bands with ImageDraw.line()            │
│  ├── render_waveform_window() → builds window analysis, then renders        │
│  ├── prerender_full_waveform() → full-track render for caching              │
│  └── crop_prerendered_image() → fast pixel crop from cache                  │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │ uses
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             analysis.py                                     │
│  WaveformAnalysis: 3-band waveform data (low/mid/high arrays)               │
│  ├── analysis_from_rb_waveform() → converts PWV N×3 array to WaveformAnalysis│
│  └── resolve_audio_path() → resolves Rekordbox PPTH to local file           │
└─────────────────────────────────────────────────────────────────────────────┘
                           ▲
                           │ loads from
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ANLZ.py                                        │
│  Rekordbox ANLZ file parsing (via pyrekordbox)                              │
│  ├── analyze_anlz_folder() → AnlzAnalysisResult (waveform, path, duration)  │
│  ├── extract_beat_grid() → List[BeatGridEntry] from PQTZ tag                │
│  └── extract_cues() → hot cues from PCOB tag (DEBUG_CUES gated)             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## GUI Controls → Config Objects

| GUI Element | Variable | Config Updated | Effect |
|-------------|----------|----------------|--------|
| **Overview Mode** checkbox | `overview_var` | `color_cfg.overview_mode` | Stacked bottom-up render vs symmetric |
| **Stack Bands** checkbox | `stack_bands_var` | `color_cfg.stack_bands` | Horizontal lanes per band |
| **Beat Grid** checkbox | `beat_grid_var` | (render flag) | Show/hide downbeat markers |
| **Zoom** slider | `zoom_var` | computed `zoom_seconds` | Window duration (0.5s – 120s) |
| **Render Mode** dropdown | `render_mode_var` | `render_cfg.render_mode` | Image height: 32/63/127px |
| **Band Order** entry | `band_order_var` | `color_cfg.band_order` | Draw order (back→front) |
| **Smoothing** slider | `smoothing_var` | `render_cfg.smoothing_bins` or `overview_smoothing_bins` | Moving average window |
| **Low/Mid/High Gain** sliders | `*_gain_var` | `render_cfg.*_gain` or `overview_*_gain` | Per-band amplitude multiplier |

### Zoom Calculation

```python
NUM_ZOOM_STEPS = 15
ZOOM_MIN = 0.5      # seconds
ZOOM_MAX = 120.0    # seconds

# Exponential curve for perceptual linearity
frac = zoom_var / NUM_ZOOM_STEPS
zoom_seconds = ZOOM_MIN * (ZOOM_MAX / ZOOM_MIN) ** frac
```

---

## Data Flow: Load Track

```
1. OSC message arrives: deck_idx, anlz_folder_path
   │
   └─► rkbx_wave._on_link_event()
       │
       └─► DeckController.load_anlz(folder)
           │
           ├─► ANLZ.analyze_anlz_folder(folder)
           │   ├─► Parse .DAT/.EXT/.2EX files (pyrekordbox.AnlzFile)
           │   ├─► Extract PWV7 tag → N×3 uint8 waveform array
           │   ├─► Extract PPTH tag → raw audio file path
           │   └─► Return AnlzAnalysisResult(waveform, song_path, duration)
           │
           ├─► analysis_from_rb_waveform(waveform, duration)
           │   ├─► Normalize each column to 0.0–1.0
           │   └─► Return WaveformAnalysis(low, mid, high, duration, ...)
           │
           └─► ANLZ.extract_beat_grid(folder)
               └─► Parse PQTZ tag → List[BeatGridEntry(time_ms, beat_number, bpm)]
```

---

## Data Flow: Render Frame

```
2. OSC message arrives: deck_idx, current_time, live_bpm
   │
   └─► rkbx_wave._on_link_event()
       │
       └─► DeckController.update_time(t), update_live_bpm(bpm)
       │
       └─► rkbx_wave._render_deck(deck_idx)
           │
           └─► DeckController.render(preview_width, preview_height, zoom_seconds, ...)
               │
               ├─► compute_timing_info(analysis)
               │   └─► TimingInfo(total_duration, n_bins, seconds_per_bin)
               │
               ├─► compute_window_plan(timing, zoom, link_time, link_scale)
               │   └─► WindowPlan(start_bin, window_bins, playhead_fraction, ...)
               │
               ├─► render_window_image(analysis, plan, cache, ...)
               │   │
               │   │  [CACHE PATH - if full prerender exists]
               │   ├─► crop_prerendered_image(cache.image, ...)
               │   │   └─► PIL.Image.crop() → fast pixel slice
               │   │
               │   │  [LIVE PATH - if cache miss or zoom changed]
               │   └─► ensure_prerender_cache(analysis, config, ...)
               │       └─► prerender_full_waveform(analysis, ...)
               │           └─► render_waveform_window(analysis, 0, n_bins, ...)
               │               └─► render_waveform_image(window_analysis, ...)
               │                   │
               │                   ├─► Build column index arrays (np.linspace)
               │                   ├─► _average_columns() per band
               │                   ├─► ImageDraw.line() per column per band
               │                   └─► Optional: beat grid lines + circles
               │
               └─► finalize_preview_image(img, target_width, target_height, ...)
                   ├─► PIL.Image.resize(interpolation) → scale to canvas
                   └─► draw_playhead_line() → vertical red line at playhead_fraction
```

---

## Libraries Used

| Library | Purpose | Key Usage |
|---------|---------|-----------|
| **PIL/Pillow** | Image rendering | `Image.new()`, `ImageDraw.line()`, `Image.resize()`, `Image.crop()` |
| **numpy** | Array operations | Band extraction, smoothing convolution, gain multiplication |
| **pyrekordbox** | ANLZ parsing | `AnlzFile.parse_file()`, tag iteration (PWV7, PPTH, PQTZ, PCOB) |
| **pythonosc** | OSC server | Receives messages from Rekordbox Link |
| **tkinter** | GUI framework | Window, Canvas, ttk widgets |

---

## Key Data Structures

### WaveformAnalysis (analysis.py)
```python
@dataclass
class WaveformAnalysis:
    low: np.ndarray      # [0..1] float32, N bins
    mid: np.ndarray      # [0..1] float32, N bins
    high: np.ndarray     # [0..1] float32, N bins
    duration_seconds: float
    sample_rate: int     # 0 for PWV-sourced data
    config_version: int  # 0 for PWV-sourced data
```

### WindowPlan (playhead.py)
```python
@dataclass
class WindowPlan:
    start_bin: int           # First visible bin
    window_bins: int         # Number of bins in view
    start_time: float        # Start time in seconds
    window_duration: float   # Duration of view in seconds
    playhead_fraction: float # 0.0–1.0 position of playhead in view
    scaled_seconds_per_bin: float  # Adjusted for BPM scaling
```

### PrerenderCache (playhead.py)
```python
@dataclass
class PrerenderCache:
    image: Image.Image        # Full-track prerendered waveform
    zoom_fraction: float      # Zoom level when rendered
    link_scale: float         # BPM scale factor
    color_cfg_hash: int       # Config fingerprint for invalidation
    render_cfg_hash: int
    analysis_id: int
```

---

## Debugging Tips

### 1. Track Not Loading
- Check ANLZ folder exists and contains `.DAT`/`.EXT`/`.2EX` files
- Verify `pyrekordbox` can parse the ANLZ file:
  ```python
  from pyrekordbox.anlz import AnlzFile
  anlz = AnlzFile.parse_file(path_to_dat)
  print([type(t).__name__ for t in anlz.tags])
  ```
- Look for `PWV7` tag (high-res waveform)

### 2. Waveform Rendering Issues
- Check `analysis.low/mid/high` arrays are non-empty and in range [0, 1]
- Verify `render_cfg.image_height` matches `RenderMode.get_render_height()`
- Test render in isolation:
  ```python
  from rb_waveform_core.render import render_waveform_image
  img = render_waveform_image(analysis, color_cfg, render_cfg)
  img.show()
  ```

### 3. Beat Grid Not Showing
- Enable `DEBUG_CUES = True` in `ANLZ.py` to see tag parsing
- Verify `extract_beat_grid()` returns non-empty list
- Check `beat_grid_var` is True in GUI

### 4. Performance / Stuttering
- Prerender cache should handle most frames → check cache invalidation
- Large `smoothing_bins` values increase render time
- Canvas resize triggers re-render → minimize window resizing

### 5. OSC Connection
- Default port: 9000
- Check `rkbx_link_listener.py` for OSC address patterns
- Test with `rkbx_link_dump.py` to see raw messages

---

## File Reference

| File | Layer | Responsibility |
|------|-------|----------------|
| `rkbx_wave.py` | GUI | Main app, event loop, canvas display |
| `rb_waveform_core/deck_controller.py` | Controller | Per-deck state, render orchestration |
| `rb_waveform_core/playhead.py` | View | Timing math, window planning, caching |
| `rb_waveform_core/render.py` | View | PIL image generation, band drawing |
| `rb_waveform_core/analysis.py` | Model | WaveformAnalysis dataclass, PWV conversion |
| `rb_waveform_core/ANLZ.py` | Model | Rekordbox file parsing (pyrekordbox) |
| `rb_waveform_core/config.py` | Config | Color/Render config dataclasses |
| `rb_waveform_core/rkbx_link_listener.py` | I/O | OSC server for Rekordbox Link |
| `rb_waveform_core/cache.py` | Util | Config persistence (JSON) |

---

## Config Persistence

- **Default config**: `default_config.json` (bundled)
- **User config**: `%APPDATA%/rkbx_wave/` or `~/.config/rkbx_wave/`
- **Last-used path**: `last_config.txt`
- Format: JSON with `color_cfg` and `render_cfg` sections
