# rkbx_wave

Real-time, zoomable waveform display for Rekordbox — with zoom levels far beyond the standard few beats, hot-cue markers, beat grid, and 2/4 deck support.

![Windows Only](https://img.shields.io/badge/platform-Windows-blue)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-green)
![License MIT](https://img.shields.io/badge/license-MIT-lightgrey)

<img width="1685" height="885" alt="1" src="https://github.com/user-attachments/assets/0c474fbf-998c-4223-9e6e-f10f381998a3" />

rkbx_wave reads Rekordbox's playback state **directly from memory** — no extra
helper program to install or configure. Open Rekordbox, run `rkbx_wave`, and a
scrolling waveform window follows your decks live.

## Features

- 2 or 4 deck waveform display, synced with Rekordbox in real time
- Zoom levels from 16 to 256 seconds (far beyond Rekordbox's on-screen waveform)
- Global zoom hotkeys (numpad +/- by default, reassignable) that work even while
  Rekordbox is the focused window
- Hot-cue and memory-cue markers (A–H colour-coded, read from your Rekordbox
  library, or from an exported USB stick)
- Beat grid overlay with adjustable line width, colour, and Rekordbox-style end markers
- RGB / overview / stacked band display modes with per-band gain and smoothing
- Always-on-top, borderless overlay mode
- Save / load configuration profiles

## Requirements

- **Windows 10 / 11** (Rekordbox memory access is Windows-only)
- **Python 3.9 or newer** — get it from [python.org](https://www.python.org/downloads/).
  During install, tick **“Add Python to PATH.”**
- **Rekordbox 7.2.2** is supported out of the box.
  [How to install a specific Rekordbox 7 version](https://rekordbox.com/en/support/faq/v7/)
  (other versions need a one-time offset file — see *Other Rekordbox versions* below).

## Installation

You don't need any coding experience — just copy/paste the commands below.

**1. Open a terminal** (press <kbd>Win</kbd>, type **PowerShell**, hit Enter).

**2. (Recommended) Create a private environment** so this doesn't touch the rest
of your system:

```powershell
python -m venv venv_rkbx
.\venv_rkbx\Scripts\Activate.ps1
```

You'll see `(venv_rkbx)` at the start of the line. To leave it later, type `deactivate`.

> If PowerShell blocks the activate script, run this once, then retry:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**3. Install rkbx_wave** with pip:

```powershell
pip install https://github.com/mrmilbe/rkbx_wave/releases/latest/download/rkbx_wave-1.1.0-py3-none-any.whl
```

Or install the latest code straight from GitHub:

```powershell
pip install git+https://github.com/mrmilbe/rkbx_wave.git
```

**4. Run it:**

```powershell
rkbx_wave
```

## Usage

1. **Start Rekordbox** and load a track onto a deck.
2. Run `rkbx_wave` (in the same terminal / environment you installed it in).
3. The status line shows **“RB Memory: connected”** once it finds Rekordbox; the
   waveform window appears and follows playback.

### Controls

| Control | Description |
|---------|-------------|
| **Zoom** | Visible time window (16–256 s). Also numpad **+ / −** while Rekordbox is focused. |
| **Decks** | Switch between 2 and 4 decks. |
| **Overview / Stack / RGB** | Waveform display modes (RGB on by default). |
| **Beat Grid** | Show bar/downbeat markers. |
| **Cues** | Show hot cues (A–H) + memory cues. |
| **On Top / Border** | Keep the waveform window above others / toggle its window border. |
| **Tune** | Open the tuning panel (gains, smoothing, beat-grid style, zoom keys). |
| **Load / Save** | Load or save a configuration profile. |

### Tuning panel

- **Band gains & smoothing** — shape the waveform; separate values for Overview mode.
- **Beat grid** — line width, colour, and optional end markers with their own colour.
- **Zoom keys** — reassign the global zoom-in / zoom-out keys (click, then press a key).

### Hot cues & memory cues

Cues come from your Rekordbox **library database** for local tracks, and from the
**ANLZ files on the stick** for tracks played off an exported Pioneer USB drive.
Hot cues are colour-coded by slot (A red, B turquoise, C green, D purple, E bright
green, F orange, G blue, H yellow); memory cues are amber.

## Configuration

Your settings live in `%APPDATA%\rkbx_wave\`:

| File | Purpose |
|------|---------|
| `default_config.json` | Copied from the package on first run; loaded at startup. |
| `last_config.txt` | Remembers the last config file you loaded. |

Tip: rather than editing `default_config.json`, use **Save Config** to create your
own profile — the last one you load is reopened automatically next time.

## Other Rekordbox versions

The app finds the right memory layout by detecting your Rekordbox version and
looking it up in `rb_waveform_core/data/offsets`. Version **7.2.2** ships built-in.
If you run a different version and cues/positions don't show, a helper tool in
[`rb_offset_detect/`](rb_offset_detect/) walks you through detecting the offsets
for your version and appending them to that file.

## Troubleshooting

**Status stays on “RB Memory: waiting…”**
- Make sure Rekordbox is actually running.
- Confirm your Rekordbox version matches an entry in the offsets file (7.2.2 by default).

**No waveform / no track loaded**
- Load a track onto a deck in Rekordbox first.
- The track must be analysed by Rekordbox (it normally is on import).

**`pip` or `python` “not recognised”**
- Reinstall Python from [python.org](https://www.python.org/downloads/) and tick
  **“Add Python to PATH.”** Then close and reopen the terminal.

## License

MIT License — see [LICENSE](LICENSE).

## Credits

- [pyrekordbox](https://github.com/dylanljones/pyrekordbox) — ANLZ / database parsing
- Memory-offset format and approach inspired by [rkbx_link](https://github.com/grufkork/rkbx_link)
