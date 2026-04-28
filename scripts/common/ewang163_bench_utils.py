"""
ewang163_bench_utils.py
=======================
Runtime benchmarking utilities for the PTSD NLP pipeline.

Provides a context manager that captures wall-clock time, CPU time, and peak
memory for any pipeline stage, appending one row per invocation to a shared
CSV log.  Designed to be imported from any script in the project.

Usage:
    from scripts.common.ewang163_bench_utils import BenchmarkLogger

    bench = BenchmarkLogger()

    with bench.track('train_longformer', stage='epoch_1', device='gpu',
                     n_samples=18264):
        # ... expensive work ...
        pass

    # Row is auto-appended to results/metrics/ewang163_runtime_benchmarks.csv

CSV columns:
    timestamp, script, stage, wall_clock_s, cpu_s, peak_mem_gb,
    n_samples, device, gpu_hours, notes
"""

import csv
import os
import resource
import time
from contextlib import contextmanager
from datetime import datetime

STUDENT_DIR = '/oscar/data/class/biol1595_2595/students/ewang163'
BENCH_CSV = f'{STUDENT_DIR}/results/metrics/ewang163_runtime_benchmarks.csv'

_CSV_HEADER = [
    'timestamp', 'script', 'stage', 'wall_clock_s', 'cpu_s',
    'peak_mem_gb', 'n_samples', 'device', 'gpu_hours', 'notes',
]


class BenchmarkLogger:
    """Append-only logger for pipeline runtime benchmarks."""

    def __init__(self, csv_path=BENCH_CSV):
        self.csv_path = csv_path
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(_CSV_HEADER)

    @contextmanager
    def track(self, script, stage='full', device='cpu', n_samples=0,
              notes=''):
        """Context manager that records one benchmark row on exit.

        Args:
            script:    Name of the script/model (e.g., 'train_longformer')
            stage:     Sub-stage label (e.g., 'epoch_3', 'inference', 'full')
            device:    'cpu' or 'gpu'
            n_samples: Number of samples processed
            notes:     Free-text annotation
        """
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        yield

        wall_s = time.perf_counter() - wall_start
        cpu_s = time.process_time() - cpu_start
        peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_kb = max(peak_before, peak_after)
        peak_gb = round(peak_kb / (1024 ** 2), 3)

        gpu_hours = round(wall_s / 3600, 4) if device == 'gpu' else 0.0

        row = [
            datetime.now().isoformat(timespec='seconds'),
            script,
            stage,
            round(wall_s, 2),
            round(cpu_s, 2),
            peak_gb,
            n_samples,
            device,
            gpu_hours,
            notes,
        ]

        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        print(f'  [BENCH] {script}/{stage}: {wall_s:.1f}s wall, '
              f'{cpu_s:.1f}s cpu, {peak_gb:.2f} GB peak'
              + (f', {gpu_hours:.4f} GPU-h' if device == 'gpu' else ''))


def format_duration(seconds):
    """Human-readable duration string."""
    if seconds < 60:
        return f'{seconds:.1f}s'
    if seconds < 3600:
        return f'{seconds / 60:.1f}m'
    return f'{seconds / 3600:.1f}h'
