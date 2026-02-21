#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

import yaml

import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _flatten_tasks(task_groups: list[dict]) -> list[str]:
    tasks = []
    for group in task_groups:
        tasks.extend(group.get("ids", []))
    return tasks


def _build_lm_eval_cmd(
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


def _parse_energy_joules(stdout: str) -> float | None:
    match = re.search(r"Energy consumption in joules:\s*([0-9.]+)", stdout)
    if not match:
        return None
    return float(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prompt-energy evaluations.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--python", default="python3")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_config(config_path)

    project = config.get("project", {})
    output_dir = Path(args.output or project.get("output_dir", "outputs"))
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
    repeats = int(runner_cfg.get("repeats", 1))

    energy_cfg = config.get("energy", {})
    energy_jsonl = Path(energy_cfg.get("output_jsonl", output_dir / "energy_runs.jsonl"))
    energy_jsonl.parent.mkdir(parents=True, exist_ok=True)
    max_execution = energy_cfg.get("max_execution")

    models = config.get("models", [])
    conditions = [c["id"] for c in config.get("prompt_conditions", [])]
    tasks = _flatten_tasks(config.get("tasks", []))

    energi_bridge_bin = Path(
        os.environ.get(
            "ENERGI_BRIDGE_BIN",
            energi_bridge_path / "target" / "release" / "energibridge",
        )
    )
    if not energi_bridge_bin.exists():
        raise SystemExit(f"EnergiBridge binary not found at {energi_bridge_bin}")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_root = output_dir / f"run_{run_id}"
    run_root.mkdir(parents=True, exist_ok=True)

    tasks_order = tasks[:]
    random.shuffle(tasks_order)

    for model in models:
        model_id = model["id"]
        backend = model.get("backend", "hf")
        trust_remote_code = bool(model.get("trust_remote_code", False))

        for condition in conditions:
            for task_id in tasks_order:
                for repeat in range(1, repeats + 1):
                    run_dir = run_root / _safe_name(model_id) / condition / task_id / f"r{repeat}"
                    run_dir.mkdir(parents=True, exist_ok=True)

                    lm_eval_output = run_dir / "lm_eval"
                    energy_csv = run_dir / "energy.csv"
                    command_output = run_dir / "lm_eval_output.txt"

                    num_fewshot = 1 if condition == "few_shot_1" else None

                    lm_eval_cmd = _build_lm_eval_cmd(
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

                    env = os.environ.copy()
                    env["PROMPT_CONDITION"] = condition
                    env["PYTHONPATH"] = (
                        f"{lm_eval_path}:{env.get('PYTHONPATH', '')}"
                    ).strip(":")

                    start_time = time.time()
                    proc = subprocess.run(
                        energibridge_cmd,
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    end_time = time.time()

                    energy_j = _parse_energy_joules(proc.stdout)

                    record = {
                        "run_id": run_id,
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

                    (run_dir / "energibridge_stdout.txt").write_text(
                        proc.stdout, encoding="utf-8"
                    )
                    (run_dir / "energibridge_stderr.txt").write_text(
                        proc.stderr, encoding="utf-8"
                    )

                    with energy_jsonl.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
