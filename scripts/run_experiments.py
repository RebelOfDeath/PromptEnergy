#!/usr/bin/env python3
"""
Run custom HumanEval evaluation with EnergiBridge energy measurement.
This wraps run_humaneval_custom.py with energy measurement.

Extended behavior:
- Can run a single job (--job-index) OR run all jobs (--all).
- When running multiple jobs, their execution order can be randomized (--shuffle).
- Adds a resting period between runs (default: 120s, configurable).
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_energy_joules(stdout: str) -> float | None:
    match = re.search(r"Energy consumption in joules:\s*([0-9.]+)", stdout)
    if not match:
        return None
    return float(match.group(1))


def run_one_job(
    *,
    args: argparse.Namespace,
    config: dict,
    jobs: list[dict],
    job_index: int,
    run_id: str,
    energi_bridge_bin: Path,
    energy_jsonl: Path,
    max_execution: int | None,
    run_root: Path,
) -> int:
    job = jobs[job_index]

    # Extract job parameters
    model_id = job["model_id"]
    condition = job["condition"]
    repeat = job["repeat"]

    # Create output directory
    run_dir = (
        run_root
        / safe_name(model_id)
        / condition
        / "humaneval_custom"
        / f"r{repeat}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    energy_csv = run_dir / "energy.csv"
    command_output = run_dir / "eval_output.txt"

    # Build the evaluation command
    script_dir = Path(__file__).parent
    eval_script = script_dir / "run_humaneval_custom.py"

    eval_cmd = [
        args.python,
        str(eval_script),
        "--config",
        args.config,
        "--jobs-file",
        args.jobs_file,
        "--job-index",
        str(job_index),
        "--run-id",
        run_id,
        "--no-energy",  # We're wrapping with energibridge
    ]

    if args.limit:
        eval_cmd.extend(["--limit", str(args.limit)])
    if args.max_new_tokens:
        eval_cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])

    # Build energibridge command
    energibridge_cmd = [
        str(energi_bridge_bin),
        "--gpu",
        "--summary",
        "--output",
        str(energy_csv),
        "--command-output",
        str(command_output),
    ]
    if max_execution is not None:
        energibridge_cmd.extend(["--max-execution", str(max_execution)])
    energibridge_cmd.extend(eval_cmd)

    # Set environment
    env = os.environ.copy()
    env["PROMPT_CONDITION"] = condition

    # Run
    print(f"Running job {job_index}: {model_id} | {condition} | repeat {repeat}")
    print(f"Output directory: {run_dir}")

    start_time = time.time()
    proc = subprocess.run(
        energibridge_cmd,
        env=env,
        capture_output=True,
        text=True,
    )
    end_time = time.time()

    energy_j = parse_energy_joules(proc.stdout)

    # Save outputs
    (run_dir / "energibridge_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (run_dir / "energibridge_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")

    # Load metrics from the inner evaluation
    summary_file = run_dir / "summary.json"
    metrics = {}
    if summary_file.exists():
        with summary_file.open("r", encoding="utf-8") as f:
            summary = json.load(f)
            metrics = summary.get("metrics", {})

    # Write record
    record = {
        "run_id": run_id,
        "job_index": job_index,
        "model_id": model_id,
        "task_id": "humaneval_custom",
        "condition_id": condition,
        "repeat": repeat,
        "exit_code": proc.returncode,
        "start_time": start_time,
        "end_time": end_time,
        "duration_s": round(end_time - start_time, 3),
        "energy_j": energy_j,
        "energy_csv": str(energy_csv),
        "output_dir": str(run_dir),
        "metrics": metrics,
        "command": energibridge_cmd,
    }

    with energy_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    print(f"Job {job_index} completed with exit code {proc.returncode}")
    if energy_j is not None:
        print(f"Energy consumption: {energy_j:.2f} J")
    if metrics:
        print("Metrics:")
        for key, value in metrics.items():
            try:
                print(f"  {key}: {float(value):.4f}")
            except (TypeError, ValueError):
                print(f"  {key}: {value}")

    return proc.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run custom HumanEval with energy measurement"
    )
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--jobs-file", default="jobs.json")

    # Single-job mode vs multi-job mode
    parser.add_argument(
        "--job-index",
        type=int,
        default=None,
        help="Run a single job by index (default behavior if --all is not provided).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all jobs from --jobs-file (optionally shuffled).",
    )

    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)

    # Randomization / resting controls
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="When running multiple jobs, randomize the execution order.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Seed for shuffling jobs. If omitted, uses runner.seed from config, else a time-based seed.",
    )
    parser.add_argument(
        "--rest-seconds",
        type=float,
        default=60.0,
        help="Resting period between job runs (seconds). Default: 120s.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned job order and exit without running anything.",
    )

    args = parser.parse_args()

    config = load_config(Path(args.config))

    # Load jobs
    with open(args.jobs_file, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"No jobs found in {args.jobs_file}")

    # Validate selection mode
    if args.all and args.job_index is not None:
        raise ValueError("Use either --all OR --job-index, not both.")
    if not args.all and args.job_index is None:
        raise ValueError("Specify --job-index to run a single job, or --all to run all jobs.")

    if args.job_index is not None and (args.job_index < 0 or args.job_index >= len(jobs)):
        raise ValueError(f"Job index {args.job_index} out of range (max {len(jobs)-1})")

    # Extract config
    project = config.get("project", {})
    output_dir = Path(project.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    runner_cfg = config.get("runner", {})
    energi_bridge_path = Path(runner_cfg.get("energi_bridge_path", "EnergiBridge"))

    energy_cfg = config.get("energy", {})
    energy_jsonl = Path(energy_cfg.get("output_jsonl", output_dir / "energy_runs.jsonl"))
    energy_jsonl.parent.mkdir(parents=True, exist_ok=True)
    max_execution = energy_cfg.get("max_execution")

    energi_bridge_bin = Path(
        os.environ.get(
            "ENERGI_BRIDGE_BIN",
            energi_bridge_path / "target" / "release" / "energibridge.exe",
        )
    )
    if not energi_bridge_bin.exists():
        raise SystemExit(f"EnergiBridge binary not found at {energi_bridge_bin}")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_root = output_dir / f"run_{run_id}"
    run_root.mkdir(parents=True, exist_ok=True)

    # Determine job indices to run
    if args.all:
        job_indices = list(range(len(jobs)))

        if args.shuffle:
            seed = args.shuffle_seed
            if seed is None:
                seed = runner_cfg.get("seed")
            if seed is None:
                seed = int(time.time())
            rng = random.Random(int(seed))
            rng.shuffle(job_indices)
            print(f"Shuffling enabled. shuffle_seed={seed}")
        else:
            print("Shuffling disabled. Running jobs in file order.")
    else:
        job_indices = [args.job_index]

    print(f"Planned runs: {len(job_indices)} job(s)")
    print(f"Rest between runs: {args.rest_seconds:.1f}s")

    if args.dry_run:
        for i, idx in enumerate(job_indices, start=1):
            job = jobs[idx]
            print(
                f"{i:>3}/{len(job_indices)} -> job_index={idx} | "
                f"{job.get('model_id')} | {job.get('condition')} | repeat {job.get('repeat')}"
            )
        return 0

    # Execute
    last_exit_code = 0
    for pos, job_index in enumerate(job_indices, start=1):
        print(f"\n=== [{pos}/{len(job_indices)}] Starting job_index={job_index} ===")
        exit_code = run_one_job(
            args=args,
            config=config,
            jobs=jobs,
            job_index=job_index,
            run_id=run_id,
            energi_bridge_bin=energi_bridge_bin,
            energy_jsonl=energy_jsonl,
            max_execution=max_execution,
            run_root=run_root,
        )
        last_exit_code = exit_code

        # Rest between runs (but not after the last one)
        if pos < len(job_indices) and args.rest_seconds > 0:
            print(f"Resting for {args.rest_seconds:.1f}s before next run...")
            time.sleep(args.rest_seconds)

    return last_exit_code


if __name__ == "__main__":
    raise SystemExit(main())