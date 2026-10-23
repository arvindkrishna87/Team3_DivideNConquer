"""Run one personal baseline split with provenance and automatic result reporting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def build_main_command(python, code, output_parent, arguments):
    """Set this process's output parent before running the unchanged baseline."""
    bootstrap = (
        f'import sys, runpy; sys.path.insert(0, {str(code)!r}); '
        f'import config; config.DATA_PATHS["runs_dir"] = {str(output_parent)!r}; '
        f'runpy.run_path({str(code / "main.py")!r}, run_name="__main__")'
    )
    return [python, '-u', '-c', bootstrap] + arguments


def run_logged_process(cmd, cwd, log):
    with log.open('x') as stream:
        process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--split', type=int, required=True)
    args = parser.parse_args()
    if args.split not in range(1, 6):
        parser.error('--split must be between 1 and 5')
    base, split = args.workspace, args.split
    code = base / 'code'
    sys.path.insert(0, str(code))
    from config import DATA_PATHS, TRAINING_CONFIG
    report = json.loads((base / 'analysis/split_validation.json').read_text())
    if report.get('all_checks_passed') is not True or report.get('seed') != 3902026:
        raise RuntimeError('Stage 3 validation/seed does not match the planned experiment')
    if TRAINING_CONFIG['random_state'] != 3902026:
        raise RuntimeError('Unexpected training seed')
    if Path(DATA_PATHS['runs_dir']).resolve() != (base / 'runs').resolve():
        raise RuntimeError('Training would write outside the personal runs directory')
    for name, digest in report['source_hashes'].items():
        if hashlib.sha256((code / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Stage 3 code changed since validation: {name}')

    split_file = base / 'splits' / f'data_splits_new_{split:02d}.npz'
    with np.load(split_file, allow_pickle=False) as data:
        counts = {k: len(data[k]) for k in ('train_cases', 'val_cases', 'test_cases')}
        with (base / 'analysis/case_assignments.csv').open() as stream:
            rows = [r for r in csv.DictReader(stream) if int(r['split']) == split]
        for key in ('train_cases', 'val_cases', 'test_cases'):
            expected = {int(r['case_id']) for r in rows if r['subset'] == key.replace('_cases', '')}
            if set(map(int, data[key])) != expected:
                raise RuntimeError(f'Input split changed after Stage 3 validation: {key}')

    job_id = os.environ.get('SLURM_JOB_ID')
    if not job_id:
        raise RuntimeError('Submit this runner through Slurm, not a login-node training command')
    out = base / 'analysis' / f'split{split:02d}'
    out.mkdir(exist_ok=False)
    log = base / 'logs' / f'training_split{split:02d}_{job_id}.log'
    output_parent = base / 'runs' / f'split{split:02d}'
    output_parent.mkdir(exist_ok=False)
    main_arguments = ['--seed', '3902026', '--epochs', '30', '--batch_size', '1',
                      '--num_workers', '2', '--per_slice_cap', '500',
                      '--max_slices_per_stain', '5', '--load_splits', str(split_file)]
    cmd = build_main_command(sys.executable, code, output_parent, main_arguments)
    manifest = {
        'status': 'RUNNING', 'job_id': job_id, 'split': split, 'seed': 3902026,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'command': cmd,
        'input_case_counts': counts, 'training_config': TRAINING_CONFIG,
        'data_paths': dict(DATA_PATHS, runs_dir=str(output_parent)),
        'main_arguments': main_arguments,
        'split_sha256': hashlib.sha256(split_file.read_bytes()).hexdigest(),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(code.glob('*.py'))},
        'python': sys.version,
        'package_versions': {p: importlib.metadata.version(p)
                             for p in ('torch', 'numpy', 'scikit-learn')},
        'cpu_cores_requested': os.environ.get('SLURM_CPUS_PER_TASK'),
        'cpu_threads': {k: os.environ.get(k) for k in
                        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        'checkpoint_policy': 'Provided baseline: final in-memory epoch; not automatically best.pth.',
        'attention_analysis': False, 'main_log': str(log),
        'array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
        'array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'launch_policy': 'Process-local config override isolates each split under runs/splitNN/; baseline source files and training settings are unchanged.',
    }
    manifest_file = out / 'run_manifest.json'
    manifest_file.write_text(json.dumps(manifest, indent=2))
    started = time.monotonic()
    try:
        returncode = run_logged_process(cmd, code, log)
        if returncode:
            raise RuntimeError(f'main.py exited with status {returncode}; see {log}')
        manifest['main_wrapper_seconds'] = time.monotonic() - started
        from report_training import summarize
        metrics = summarize(base, split, log, out)
        manifest['status'] = 'COMPLETED'
        manifest['run_directory'] = metrics['run_directory']
        manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
    except BaseException as error:
        manifest['status'] = 'FAILED'
        manifest['error'] = str(error)
        raise
    finally:
        manifest_file.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
