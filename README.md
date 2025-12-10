# rkbx_wave

Real-time waveform display for Rekordbox DJ with zoom levels beyond the standard 6 beats.

![Windows Only](https://img.shields.io/badge/platform-Windows-blue)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-green)
![License MIT](https://img.shields.io/badge/license-MIT-lightgrey)

<img width="1685" height="885" alt="1" src="https://github.com/user-attachments/assets/0c474fbf-998c-4223-9e6e-f10f381998a3" />


## Features

- Dual deck waveform display synced with Rekordbox in real-time
- Configurable zoom levels from 16 to 256 seconds
- Per-band gain control (Low/Mid/High)
- Beat grid overlay
- RB style overview mode with adjustable smoothing
- Stacked view with isolated bands
- Configuration save/load with persistent settings
- Tuning / customization of waveform rendering

## TBD
- enable all RB versions
- add 4 deck support
- add cue & loop markers
- ....


## Requirements

- **Windows 10/11** (sorry apple folks :/)
- **Python 3.9+**
- **Rekordbox 7.2.2** 
([click on "I want to use previous rekordbox ver. 7." in the FAQ](https://rekordbox.com/en/support/faq/v7/))

## Installation

### From GitHub Releases (Recommended)

1. (Optional but Highly Recommended) Use a Virtual Environment


**Create and activate a venv:**

Windows PowerShell:
```powershell
python -m venv venv_rkbx
.\venv_rkbx\Scripts\Activate.ps1
```

Windows Command Prompt:
```cmd
python -m venv venv_rkbx
venv_rkbx\Scripts\activate.bat
```

To deactivate later, just run:
```
deactivate
```

2. Install with pip ( inside your venv):
   ```
   pip install https://github.com/mrmilbe/rkbx_wave/releases/download/pre-release/rkbx_wave-1.0.0-py3-none-any.whl
   ```
3. Run:
   ```
   rkbx_wave
   ```


## Usage

### Starting the Application

1. **Start Rekordbox** first and load tracks onto decks
2. Run `rkbx_wave` from command line or terminal
3. The application will automatically connect to Rekordbox via rkbx_link

### Main Controls

| Control | Description |
|---------|-------------|
| **Tune** | Toggle the tuning panel (left side) |
| **Load Config** | Load a saved configuration file |
| **Save Config** | Save current settings to a file |
| **Overview Mode** | Switch between detailed and overview waveform |
| **Stack Bands** | Stack frequency bands vertically |
| **Beat Grid** | Show/hide beat markers |
| **Zoom** | Adjust visible time window (16-256 seconds) |

### Tuning Panel Settings

#### Render Mode
- **Speed** - Fastest rendering, lower detail (32px height)
- **Default** - Balanced performance and quality (63px height)  
- **Quality** - Best visual quality, slower rendering (127px height)

#### Band Order
Configure the vertical stacking order of frequency bands:
- `l,m,h` - Low on bottom, High on top (default)
- `h,m,l` - High on bottom, Low on top (overview default)
- Any combination of `l` (low), `m` (mid), `h` (high)

#### Smoothing
Adjusts waveform smoothing (1-63 bins). Higher values create smoother waveforms, useful for overview mode.

#### Band Gains
Adjust the visual intensity of each frequency band:
- **Low** - Bass frequencies (blue by default)
- **Mid** - Midrange frequencies (orange by default)
- **High** - High frequencies (white by default)

Range: 0.0 to 3.0 (1.0 = no change)

## Configuration Files

Settings are stored in `%APPDATA%\rkbx_wave\`:

| File | Purpose |
|------|---------|
| `default_config.json` | Default settings (copied on first run) |
| `last_config.txt` | Path to last loaded config file |

I recomend not to change `default_config.json`, you can save a new config in a different file and it will be loaded on startup.

## Troubleshooting

### "rkbx_link.exe not found"
The executable should be installed automatically. If missing, reinstall the package.

### No waveforms showing
- Ensure Rekordbox is running with tracks loaded
- Check that rkbx_link.exe is not blocked by firewall/antivirus
- Verify Rekordbox version is 7.2.2

### High CPU usage
- Switch to **Speed** render mode
- Reduce window width
- Close other resource-intensive applications

## License

MIT License - see [LICENSE](LICENSE) for details.

## Credits

- Uses [pyrekordbox](https://github.com/dylanljones/pyrekordbox) for ANLZ file parsing
- Uses [rkbx_link](https://github.com/grufkork/rkbx_link) for real-time Rekordbox communication
