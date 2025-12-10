# rb_waveform_lab

Experimental and debugging utilities for waveform analysis. **Not included in production builds.**

## Purpose

This package contains:
- **Audio analysis pipeline** (`analysis.py`): Four-band spectral analysis with filtering, compression, and normalization
- **Live rendering helpers** (`antialias.py`): Antialiasing cache for live waveform display
- **Tuning GUI** (`tuning_gui.py`): Interactive parameter tuning interface

## Usage

Lab tools are for internal experimentation and parameter tuning only. Production code (`waveformsync_gui.py`) uses Rekordbox PWV data via `rb_waveform_core`.

### Running the Tuning GUI

```powershell
python -m rb_waveform_lab.tuning_gui
```

### Configuration

Lab-specific config: `waveform_lab_config.json`

## Package Split

- **rb_waveform_core**: Production code (PWV conversion, ANLZ readers, beat grid, rendering, timing)
- **rb_waveform_lab**: Experimental code (audio analysis, live render, tuning tools)

Production builds should exclude `rb_waveform_lab/` to keep the distribution lean.
