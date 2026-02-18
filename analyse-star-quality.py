#!/usr/bin/python3
"""
Analyse star quality metrics (HFR, elongation, SNR) in astrophotography raw images.

Usage:
    # Analyse images and create CSV (shell expands the glob)
    python3 analyse-star-quality.py analyse /path/to/lights/*.orf
    python3 analyse-star-quality.py analyse /path/to/lights/*.cr2 --output results.csv

    # Analyse with a glob pattern (script expands it)
    python3 analyse-star-quality.py analyse '/path/to/lights/*.orf'

    # Rename files based on existing CSV
    python3 analyse-star-quality.py rename /path/to/directory --by hfr
    python3 analyse-star-quality.py rename /path/to/directory --by quality --csv results.csv

Dependencies:
    pip install rawpy numpy sep
"""

import os
import sys
import argparse
import csv
import glob
import multiprocessing
from functools import partial
import rawpy
import numpy as np

try:
    import sep
except ImportError:
    print("Error: 'sep' library required. Install with: pip install sep")
    sys.exit(1)


def bin_2x2(data):
    """Bin raw Bayer data 2x2 to get a regular grid with preserved star shapes."""
    h, w = data.shape
    # Ensure dimensions are even
    h = h - (h % 2)
    w = w - (w % 2)
    data = data[:h, :w]
    return data.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def measure_hfr_batch(data_sub, x_arr, y_arr, max_radius=25.0):
    """
    Measure Half-Flux Radius (HFR) for multiple stars.

    HFR is the radius containing 50% of the total flux. It's more sensitive
    to defocus than FWHM because it measures actual flux distribution rather
    than just fitting the bright core shape.

    Returns array of HFR values, one per star.
    """
    n_stars = len(x_arr)
    hfr_values = np.full(n_stars, np.nan)

    # Get total flux for each star in a large aperture
    total_flux, _, _ = sep.sum_circle(data_sub, x_arr, y_arr, max_radius)

    # Test radii from 0.5 to max_radius
    test_radii = np.arange(0.5, max_radius + 0.5, 0.5)

    # Compute cumulative flux at each test radius for all stars
    # Shape will be (n_radii, n_stars)
    cumulative_flux = np.zeros((len(test_radii), n_stars))
    for i, r in enumerate(test_radii):
        flux, _, _ = sep.sum_circle(data_sub, x_arr, y_arr, r)
        cumulative_flux[i] = flux

    # For each star, find radius where flux reaches 50%
    for j in range(n_stars):
        if total_flux[j] <= 0:
            continue

        half_flux = total_flux[j] * 0.5
        flux_curve = cumulative_flux[:, j]

        # Find where we cross 50%
        idx = np.searchsorted(flux_curve, half_flux)
        if idx == 0:
            hfr_values[j] = test_radii[0]
        elif idx >= len(test_radii):
            hfr_values[j] = test_radii[-1]
        else:
            # Linear interpolation between adjacent radii
            r_low, r_high = test_radii[idx - 1], test_radii[idx]
            f_low, f_high = flux_curve[idx - 1], flux_curve[idx]
            if f_high > f_low:
                frac = (half_flux - f_low) / (f_high - f_low)
                hfr_values[j] = r_low + frac * (r_high - r_low)
            else:
                hfr_values[j] = r_low

    return hfr_values


def analyse_image(file_path, use_binning=True, debug=False):
    """
    Analyse a single raw image for star quality metrics.

    Returns dict with: hfr, elongation, snr, background, quality, stars_detected
    """
    with rawpy.imread(file_path) as raw:
        raw_data = raw.raw_image_visible.astype(np.float64)
        max_value = float(raw.white_level)  # Saturation level

    # Measure background as simple mean of raw data (matches visual impression)
    background = np.mean(raw_data)

    # Optionally bin 2x2 (faster, but less precision)
    if use_binning:
        data = bin_2x2(raw_data)
    else:
        data = raw_data

    # Ensure C-contiguous array for sep
    data = np.ascontiguousarray(data)

    # Background estimation and subtraction for star detection
    bkg = sep.Background(data)
    data_sub = data - bkg

    # Extract sources with larger minimum area to avoid hot pixels
    try:
        objects = sep.extract(data_sub, thresh=5.0, err=bkg.globalrms, minarea=9)
    except Exception as e:
        return {
            'hfr': float('nan'),
            'elongation': float('nan'),
            'snr': float('nan'),
            'background': background,
            'quality': float('nan'),
            'stars_detected': 0,
            'error': str(e)
        }

    if debug:
        print(f"\n    Raw detections: {len(objects)}")

    if len(objects) == 0:
        return {
            'hfr': float('nan'),
            'elongation': float('nan'),
            'snr': float('nan'),
            'background': background,
            'quality': float('nan'),
            'stars_detected': 0,
            'error': None
        }

    # Calculate approximate FWHM for initial filtering (hot pixel rejection)
    fwhm_approx = 2.355 * np.sqrt(objects['a'] * objects['b'])
    snr_all = objects['peak'] / bkg.globalrms

    # Filter to get real stars (but don't filter by shape - we want to measure that)
    margin = 25  # Need margin for HFR aperture photometry
    h, w = data.shape
    saturation_thresh = 0.9 * max_value - bkg.globalback  # Avoid saturated stars

    mask = (
        # Not at edges (larger margin for HFR measurement)
        (objects['x'] > margin) & (objects['x'] < w - margin) &
        (objects['y'] > margin) & (objects['y'] < h - margin) &
        # Size filters: real stars should have FWHM > 2 pixels (not hot pixels)
        (fwhm_approx > 2.0) &
        (fwhm_approx < 30.0) &  # Not too large (galaxies, artifacts)
        # Not saturated
        (objects['peak'] < saturation_thresh) &
        # High SNR - real stars should be well above noise
        (snr_all > 20)
    )
    # Note: deliberately NOT filtering by roundness - we want to measure elongation

    if debug:
        print(f"    After edge filter: {np.sum((objects['x'] > margin) & (objects['x'] < w - margin))}")
        print(f"    FWHM_approx > 2.0: {np.sum(fwhm_approx > 2.0)}")
        print(f"    SNR > 20: {np.sum(snr_all > 20)}")
        print(f"    Not saturated: {np.sum(objects['peak'] < saturation_thresh)}")
        print(f"    After all filters: {np.sum(mask)}")

    objects = objects[mask]
    snr_values = snr_all[mask]

    if len(objects) == 0:
        return {
            'hfr': float('nan'),
            'elongation': float('nan'),
            'snr': float('nan'),
            'background': background,
            'quality': float('nan'),
            'stars_detected': 0,
            'error': 'No stars passed filters (try --debug)'
        }

    # Use only the BRIGHTEST stars for quality metrics
    # Bright stars show aberrations clearly; faint stars hide them in noise
    # This makes the metric match visual impression much better
    n_bright = min(30, len(objects))
    brightness_order = np.argsort(objects['peak'])[::-1]  # Brightest first
    bright_idx = brightness_order[:n_bright]

    bright_objects = objects[bright_idx]
    bright_snr = snr_values[bright_idx]

    if debug:
        print(f"    Using top {n_bright} brightest stars for metrics")
        print(f"    Brightness range: {objects['peak'][bright_idx].min():.0f} - {objects['peak'][bright_idx].max():.0f}")

    # Measure HFR using aperture photometry (more sensitive to defocus than FWHM)
    hfr_values = measure_hfr_batch(data_sub, bright_objects['x'], bright_objects['y'])

    # Compute elongation from sep's shape parameters
    minor_axis = np.minimum(bright_objects['a'], bright_objects['b'])
    major_axis = np.maximum(bright_objects['a'], bright_objects['b'])
    elongation_values = major_axis / minor_axis
    snr_values = bright_snr

    # Filter out stars with invalid HFR
    valid_hfr = ~np.isnan(hfr_values)
    if np.sum(valid_hfr) == 0:
        return {
            'hfr': float('nan'),
            'elongation': float('nan'),
            'snr': float('nan'),
            'background': background,
            'quality': float('nan'),
            'stars_detected': 0,
            'error': 'No valid HFR measurements'
        }

    hfr_values = hfr_values[valid_hfr]
    elongation_values = elongation_values[valid_hfr]
    snr_values = snr_values[valid_hfr]

    if debug:
        print(f"    Final stars with valid HFR: {len(hfr_values)}")
        print(f"    HFR range: {hfr_values.min():.2f} - {hfr_values.max():.2f}")
        print(f"    HFR 90th pct: {np.percentile(hfr_values, 90):.2f}")
        print(f"    Elongation range: {elongation_values.min():.2f} - {elongation_values.max():.2f}")

    # Use 90th percentile for HFR to capture edge aberrations (median misses them)
    # Use median for elongation and SNR (these are more uniform across the field)
    hfr = np.percentile(hfr_values, 90)
    elongation = np.median(elongation_values)
    snr = np.median(snr_values)

    # Combined quality score: higher = better
    # SNR in numerator (higher is better), HFR and elongation in denominator (lower is better)
    # HFR is typically larger than FWHM_minor, so quality values will be different from before
    quality = snr / (hfr * elongation * np.sqrt(background))

    return {
        'hfr': hfr,
        'elongation': elongation,
        'snr': snr,
        'background': background,
        'quality': quality,
        'stars_detected': len(hfr_values),
        'error': None
    }


def _analyse_worker(args_tuple):
    """Worker function for multiprocessing."""
    file_path, filename, use_binning = args_tuple
    try:
        metrics = analyse_image(file_path, use_binning=use_binning, debug=False)
        metrics['filename'] = filename
        return metrics
    except Exception as e:
        return {
            'filename': filename,
            'hfr': float('nan'),
            'elongation': float('nan'),
            'snr': float('nan'),
            'background': float('nan'),
            'quality': float('nan'),
            'stars_detected': 0,
            'error': str(e)
        }


def cmd_analyse(args):
    """Analyse all images and output CSV."""
    # Expand globs (handles both shell-expanded paths and quoted glob patterns)
    file_paths = []
    for pattern in args.files:
        expanded = glob.glob(pattern)
        if expanded:
            file_paths.extend(expanded)
        elif os.path.isfile(pattern):
            file_paths.append(pattern)
        else:
            print(f"Warning: no files matched '{pattern}'")

    if not file_paths:
        print("Error: no input files found.")
        return 1

    # Deduplicate and sort
    file_paths = sorted(set(os.path.abspath(p) for p in file_paths))
    total_files = len(file_paths)

    # Determine output directory (directory of first input file)
    output_dir = os.path.dirname(file_paths[0])

    mode = "full resolution" if args.no_bin else "2x2 binned"
    num_jobs = args.jobs if not args.debug else 1  # Debug mode forces single-threaded
    if args.debug and args.jobs > 1:
        print("Note: --debug forces single-threaded mode")

    print(f"Analysing {total_files} files ({mode}, {num_jobs} workers)")

    results = []
    use_binning = not args.no_bin

    if num_jobs > 1:
        # Parallel processing
        work_items = [(fp, os.path.basename(fp), use_binning) for fp in file_paths]
        with multiprocessing.Pool(num_jobs) as pool:
            for i, metrics in enumerate(pool.imap_unordered(_analyse_worker, work_items), 1):
                if metrics['error']:
                    status = f"error: {metrics['error']}"
                elif metrics['stars_detected'] == 0:
                    status = "no stars detected"
                else:
                    status = (f"HFR={metrics['hfr']:.2f} elong={metrics['elongation']:.2f} "
                              f"Q={metrics['quality']:.3f} stars={metrics['stars_detected']}")
                print(f"[{i}/{total_files}] {metrics['filename']}... {status}")
                results.append(metrics)
    else:
        # Sequential processing (with optional debug)
        for i, file_path in enumerate(file_paths, 1):
            filename = os.path.basename(file_path)
            print(f"[{i}/{total_files}] {filename}...", end=" ", flush=True)

            try:
                metrics = analyse_image(file_path, use_binning=use_binning, debug=args.debug)
                metrics['filename'] = filename
                results.append(metrics)

                if metrics['error']:
                    print(f"error: {metrics['error']}")
                elif metrics['stars_detected'] == 0:
                    print("no stars detected")
                else:
                    print(f"HFR={metrics['hfr']:.2f} elong={metrics['elongation']:.2f} "
                          f"Q={metrics['quality']:.3f} stars={metrics['stars_detected']}")
            except Exception as e:
                print(f"error: {e}")
                results.append({
                    'filename': filename,
                    'hfr': float('nan'),
                    'elongation': float('nan'),
                    'snr': float('nan'),
                    'background': float('nan'),
                    'quality': float('nan'),
                    'stars_detected': 0,
                    'error': str(e)
                })

    # Output CSV
    output_csv = args.output
    if output_csv is None:
        output_csv = os.path.join(output_dir, "star_quality.csv")

    fieldnames = ['filename', 'hfr', 'elongation', 'snr', 'background', 'quality', 'stars_detected']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {output_csv}")

    # Print summary statistics
    valid_results = [r for r in results if r['stars_detected'] > 0]
    if valid_results:
        print(f"\nSummary ({len(valid_results)} valid frames):")
        for metric in ['hfr', 'elongation', 'snr', 'background', 'quality']:
            values = [r[metric] for r in valid_results if not np.isnan(r[metric])]
            if values:
                print(f"  {metric:12s}: min={min(values):.2f}  median={np.median(values):.2f}  max={max(values):.2f}")

    return 0


def cmd_rename(args):
    """Rename files based on existing CSV."""
    source_dir = args.directory
    metric = args.by

    if not os.path.isdir(source_dir):
        print(f"Error: {source_dir} is not a valid directory.")
        return 1

    # Find CSV file
    csv_path = args.csv
    if csv_path is None:
        csv_path = os.path.join(source_dir, "star_quality.csv")

    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        print("Run 'analyse' first to create the CSV.")
        return 1

    # Read CSV
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("Error: CSV file is empty.")
        return 1

    print(f"Renaming {len(rows)} files by {metric}...")

    renamed = 0
    skipped = 0

    for row in rows:
        filename = row['filename']
        value_str = row.get(metric, '')

        # Skip if value is missing or nan
        try:
            value = float(value_str)
            if np.isnan(value):
                raise ValueError()
        except (ValueError, TypeError):
            print(f"  Skipping {filename}: no valid {metric} value")
            skipped += 1
            continue

        old_path = os.path.join(source_dir, filename)

        # Check if file exists (might already be renamed)
        if not os.path.exists(old_path):
            print(f"  Skipping {filename}: file not found")
            skipped += 1
            continue

        # Format prefix based on metric (enough precision to differentiate)
        if metric == 'elongation':
            prefix = f"{metric}_{value:.4f}_"
        elif metric == 'snr':
            prefix = f"{metric}_{value:06.1f}_"  # Zero-padded for sorting
        elif metric == 'hfr':
            prefix = f"{metric}_{value:05.2f}_"  # Zero-padded for sorting (lower = better)
        elif metric == 'quality':
            prefix = f"{metric}_{value:07.4f}_"  # Zero-padded for sorting (higher = better)
        else:  # background
            prefix = f"{metric}_{value:07.1f}_"  # Zero-padded for sorting

        new_name = prefix + filename
        new_path = os.path.join(source_dir, new_name)

        try:
            os.rename(old_path, new_path)
            renamed += 1
        except Exception as e:
            print(f"  Error renaming {filename}: {e}")
            skipped += 1

    print(f"\nRenamed {renamed} files, skipped {skipped}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyse star quality metrics in astrophotography raw images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyse Olympus raw files
  %(prog)s analyse /path/to/lights/*.orf

  # Analyse Canon raw files with 8 parallel workers
  %(prog)s analyse /path/to/lights/*.cr2 -j 8

  # Analyse multiple formats at once
  %(prog)s analyse /path/to/lights/*.orf /path/to/lights/*.nef

  # Use a quoted glob (script expands it)
  %(prog)s analyse '/path/to/lights/*.orf'

  # Analyse with higher precision (slower)
  %(prog)s analyse /path/to/lights/*.orf --no-bin

  # Analyse with custom output file
  %(prog)s analyse /path/to/lights/*.orf --output results.csv

  # Rename files by HFR (using star_quality.csv)
  %(prog)s rename /path/to/lights --by hfr

  # Rename by quality using custom CSV
  %(prog)s rename /path/to/lights --by quality --csv results.csv

Metrics:
  hfr         Half-Flux Radius 90th percentile in pixels (lower = better)
              Uses 90th percentile to catch edge/corner aberrations that
              median would miss. Measures radius containing 50%% of star flux.
  elongation  Star stretch ratio (1.0 = round, higher = wind shake)
  snr         Signal-to-noise ratio (higher = better)
  background  Sky brightness level (lower = darker sky)
  quality     Combined score: SNR/(hfr*elong*sqrt(bg)) (higher = better)
"""
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Analyse subcommand
    analyse_parser = subparsers.add_parser('analyse', help='Analyse images and create CSV')
    analyse_parser.add_argument("files", nargs='+',
                                help="Raw image files or glob patterns (e.g. *.orf *.cr2 *.nef)")
    analyse_parser.add_argument("--output", "-o", help="Output CSV file path (default: star_quality.csv in input dir)")
    analyse_parser.add_argument("--no-bin", action="store_true",
                                help="Skip 2x2 binning for higher precision (slower)")
    analyse_parser.add_argument("--debug", action="store_true",
                                help="Show detailed detection diagnostics")
    analyse_parser.add_argument("-j", "--jobs", type=int, default=multiprocessing.cpu_count(),
                                help=f"Number of parallel workers (default: {multiprocessing.cpu_count()})")

    # Rename subcommand
    rename_parser = subparsers.add_parser('rename', help='Rename files based on existing CSV')
    rename_parser.add_argument("directory", help="Directory containing the image files")
    rename_parser.add_argument("--by", required=True,
                               choices=['hfr', 'elongation', 'snr', 'background', 'quality'],
                               help="Metric to use for renaming")
    rename_parser.add_argument("--csv", help="Path to CSV file (default: star_quality.csv in source dir)")

    args = parser.parse_args()

    if args.command == 'analyse':
        sys.exit(cmd_analyse(args))
    elif args.command == 'rename':
        sys.exit(cmd_rename(args))
