# Copyright (c) mrmilbe

"""Setup configuration for rkbx_wave package."""

from setuptools import setup, find_packages
from pathlib import Path
import glob

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Collect rkbx_link files
rkbx_link_files = []
rkbx_link_dir = Path(__file__).parent / "rkbx_link"
if rkbx_link_dir.exists():
    for f in rkbx_link_dir.rglob("*"):
        if f.is_file():
            rel_path = f.relative_to(Path(__file__).parent)
            rkbx_link_files.append(str(rel_path))

setup(
    name="rkbx_wave",
    version="1.0.0",
    author="mrmilbe",
    description="Dual deck waveform display for Rekordbox with live sync via rkbx_link",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrmilbe/rkbx_wave",
    
    # Main module and packages
    py_modules=["rkbx_wave"],
    packages=find_packages(include=["rb_waveform_core", "rb_waveform_core.*"]),
    
    # Data files installed to sys.prefix/rkbx_wave_data
    data_files=[
        ("rkbx_wave_data", ["default_config.json"]),
        ("rkbx_wave_data/rkbx_link", [
            "rkbx_link/rkbx_link.exe",
            "rkbx_link/config",
        ]),
        ("rkbx_wave_data/rkbx_link/data", ["rkbx_link/data/offsets"]),
    ],
    
    # Console script entry point
    entry_points={
        "console_scripts": [
            "rkbx_wave=rkbx_wave:main",
        ],
    },
    
    # Dependencies
    install_requires=[
        "Pillow>=10.0.0",
        "construct>=2.10.0",
        "mutagen>=1.47.0",
        "numpy>=1.24.0",
        "psutil>=5.9.0",
        "pyrekordbox>=0.4.0",
        "python-osc>=1.9.0",
        "scipy>=1.10.0",
    ],
    
    python_requires=">=3.9",
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: Microsoft :: Windows",
        "Environment :: Win32 (MS Windows)",
    ],
    
    # Platform restriction
    platforms=["win32"],
)
