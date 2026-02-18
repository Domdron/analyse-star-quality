# analyse-star-quality

Analyse star quality metrics (HFR, elongation, SNR) in astrophotography raw images. Designed for grading and sorting sub-exposures to identify the best frames for stacking.

## Metrics

| Metric | Description | Ideal |
|---|---|---|
| **HFR** | Half-Flux Radius (90th percentile, pixels). Radius containing 50% of star flux. Uses 90th percentile to catch edge/corner aberrations that median would miss. | Lower is better |
| **Elongation** | Star stretch ratio (major/minor axis). 1.0 = perfectly round. | Lower (closer to 1.0) |
| **SNR** | Signal-to-noise ratio of detected stars. | Higher is better |
| **Background** | Mean sky brightness level from the raw sensor data. | Lower = darker sky |
| **Quality** | Combined score: `SNR / (HFR * elongation * sqrt(background))`. | Higher is better |

The tool focuses on the **brightest 30 stars** in each frame, since bright stars reveal optical aberrations (defocus, tilt, wind shake) most clearly.

## Setup

Requires Python 3.8+.

```bash
# Clone the repository
git clone git@domdron-github.com:Domdron/analyse-star-quality.git
cd analyse-star-quality

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The tool has two subcommands: `analyse` and `rename`.

### Analyse images

Pass raw image files (or glob patterns) to measure star quality metrics and write results to a CSV file. Supports any raw format readable by [rawpy/LibRaw](https://www.libraw.org/) (`.orf`, `.cr2`, `.cr3`, `.nef`, `.arw`, `.dng`, `.rw2`, `.raf`, etc.).

```bash
# Analyse Olympus raw files (creates star_quality.csv in the input directory)
python3 analyse-star-quality.py analyse /path/to/lights/*.orf

# Analyse Canon CR2 files with 8 parallel workers
python3 analyse-star-quality.py analyse /path/to/lights/*.cr2 -j 8

# Mix multiple formats
python3 analyse-star-quality.py analyse /path/to/lights/*.orf /path/to/lights/*.nef

# Quoted glob (the script expands it instead of the shell)
python3 analyse-star-quality.py analyse '/path/to/lights/*.orf'

# Higher precision analysis without 2x2 binning (slower)
python3 analyse-star-quality.py analyse /path/to/lights/*.orf --no-bin

# Write results to a custom CSV path
python3 analyse-star-quality.py analyse /path/to/lights/*.orf --output results.csv

# Debug mode: show detailed star detection diagnostics (single-threaded)
python3 analyse-star-quality.py analyse /path/to/lights/*.orf --debug
```

### Rename files by metric

Prepend a metric value to each filename so files sort by quality in a file manager. Requires a CSV from a prior `analyse` run.

```bash
# Rename files with HFR prefix (e.g. hfr_03.45_IMG_0001.orf)
python3 analyse-star-quality.py rename /path/to/lights --by hfr

# Rename by combined quality score
python3 analyse-star-quality.py rename /path/to/lights --by quality

# Use a specific CSV file
python3 analyse-star-quality.py rename /path/to/lights --by snr --csv results.csv
```

Available metrics for `--by`: `hfr`, `elongation`, `snr`, `background`, `quality`.

## How it works

1. **Raw decoding** -- Each raw file is read with `rawpy` (a LibRaw wrapper) to access the raw Bayer sensor data. Any format supported by LibRaw works.
2. **Binning** (optional) -- The Bayer data is 2x2 binned to produce a smaller monochrome image, which speeds up processing while preserving star shapes.
3. **Background estimation** -- `sep` (Source Extractor as a Python library) estimates and subtracts the sky background.
4. **Star detection** -- Sources are extracted above 5-sigma with a minimum area of 9 pixels. Filters remove hot pixels, saturated stars, edge detections, and faint sources.
5. **HFR measurement** -- For the brightest 30 stars, Half-Flux Radius is measured via aperture photometry at increasing radii until 50% of the total flux is enclosed.
6. **Elongation** -- Computed from `sep`'s fitted ellipse parameters (major axis / minor axis).
7. **Quality score** -- A combined metric that rewards high SNR and penalises large HFR, elongation, and sky brightness.

## Supported formats

Supports any raw camera format readable by [LibRaw](https://www.libraw.org/), including: `.orf` (Olympus), `.cr2`/`.cr3` (Canon), `.nef` (Nikon), `.arw` (Sony), `.dng` (Adobe), `.rw2` (Panasonic), `.raf` (Fujifilm), `.pef` (Pentax), and more.

## License

MIT
