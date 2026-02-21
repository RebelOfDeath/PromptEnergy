#!/usr/bin/env python3
"""Run a single experiment job (used by SLURM array)."""
import argparse
import json
import os
import re
import subprocess
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


def build_lm_eval_cmd(
    python_bin: str,
    lm_eval_path: Path,
    model_backend: str,
    model_id: str,
    task_id: str,
    output_path: Path,
    include_path: Path | None,
    device: str | None,
    seed: int | None,
    num_fewshot: int | None,
    trust_remote_code: bool,
) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        "lm_eval",
        "--model",
        model_backend,
        "--model_args",
        f"pretrained={model_id}",
        "--tasks",
        task_id,
        "--output_path",
        str(output_path),
        "--confirm_run_unsafe_code",
    ]

    if include_path is not None:
        cmd.extend(["--include_path", str(include_path)])
    if device:
        cmd.extend(["--device", device])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])
    if trust_remote_code:
        cmd.append("--trust_remote_code")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run single evaluation job")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--jobs-file", default="jobs.json")
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))

    # Load jobs
    with open(args.jobs_file, "r") as f:
        jobs = json.load(f)

    if args.job_index >= len(jobs):
        raise ValueError(f"Job index {args.job_index} out of range (max {len(jobs)-1})")

    job = jobs[args.job_index]

    # Extract config
    project = config.get("project", {})
    output_dir = Path(project.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    runner_cfg = config.get("runner", {})
    lm_eval_path = Path(runner_cfg.get("lm_eval_harness_path", "lm-evaluation-harness"))
    energi_bridge_path = Path(runner_cfg.get("energi_bridge_path", "EnergiBridge"))
    include_path = runner_cfg.get("include_path")
    if include_path:
        include_path = Path(include_path)

    device = runner_cfg.get("device")
    if isinstance(device, str) and device.lower() == "auto":
        device = None
    seed = runner_cfg.get("seed")

    energy_cfg = config.get("energy", {})
    energy_jsonl = Path(energy_cfg.get("output_jsonl", output_dir / "energy_runs.jsonl"))
    energy_jsonl.parent.mkdir(parents=True, exist_ok=True)
    max_execution = energy_cfg.get("max_execution")

    energi_bridge_bin = Path(
        os.environ.get(
            "ENERGI_BRIDGE_BIN",
            energi_bridge_path / "target" / "release" / "energibridge",
        )
    )
    if not energi_bridge_bin.exists():
        raise SystemExit(f"EnergiBridge binary not found at {energi_bridge_bin}")

    # Use provided run_id or generate one
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_root = output_dir / f"run_{run_id}"
    run_root.mkdir(parents=True, exist_ok=True)

    # Extract job parameters
    model_id = job["model_id"]
    backend = job["backend"]
    trust_remote_code = bool(job["trust_remote_code"])
    condition = job["condition"]
    task_id = job["task"]
    repeat = job["repeat"]

    # Create output directory
    run_dir = run_root / safe_name(model_id) / condition / task_id / f"r{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)

    lm_eval_output = run_dir / "lm_eval"
    energy_csv = run_dir / "energy.csv"
    command_output = run_dir / "lm_eval_output.txt"

    # Handle few_shot condition
    num_fewshot = 1 if condition == "few_shot_1" else None

    # Build command
    lm_eval_cmd = build_lm_eval_cmd(
        args.python,
        lm_eval_path,
        backend,
        model_id,
        task_id,
        lm_eval_output,
        include_path,
        device,
        seed,
        num_fewshot,
        trust_remote_code,
    )

    energibridge_cmd = [
        str(energi_bridge_bin),
        "--summary",
        "--output",
        str(energy_csv),
        "--command-output",
        str(command_output),
    ]
    if max_execution is not None:
        energibridge_cmd.extend(["--max-execution", str(max_execution)])
    energibridge_cmd.extend(lm_eval_cmd)

    # Set environment
    env = os.environ.copy()
    env["PROMPT_CONDITION"] = condition
    env["PYTHONPATH"] = (
        f"{lm_eval_path}:{env.get('PYTHONPATH', '')}"
    ).strip(":")
    env["HF_ALLOW_CODE_EVAL"] = "1"

    # Run
    print(f"Running job {args.job_index}: {model_id} | {condition} | {task_id} | repeat {repeat}")
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
    (run_dir / "energibridge_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "energibridge_stderr.txt").write_text(proc.stderr, encoding="utf-8")

    # Write record
    record = {
        "run_id": run_id,
        "job_index": args.job_index,
        "model_id": model_id,
        "task_id": task_id,
        "condition_id": condition,
        "repeat": repeat,
        "backend": backend,
        "exit_code": proc.returncode,
        "start_time": start_time,
        "end_time": end_time,
        "duration_s": round(end_time - start_time, 3),
        "energy_j": energy_j,
        "energy_csv": str(energy_csv),
        "lm_eval_output_path": str(lm_eval_output),
        "command": energibridge_cmd,
        "stderr_path": str(run_dir / "energibridge_stderr.txt"),
        "stdout_path": str(run_dir / "energibridge_stdout.txt"),
    }

    with energy_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    print(f"Job {args.job_index} completed with exit code {proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
