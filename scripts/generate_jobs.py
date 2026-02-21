#!/usr/bin/env python3
"""Generate SLURM job array commands for all experiment combinations."""
import argparse
import json
from pathlib import Path
import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_tasks(task_groups: list[dict]) -> list[str]:
    tasks = []
    for group in task_groups:
        tasks.extend(group.get("ids", []))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Generate job list for SLURM array")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--output", default="jobs.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    models = config.get("models", [])
    conditions = [c["id"] for c in config.get("prompt_conditions", [])]
    tasks = flatten_tasks(config.get("tasks", []))
    repeats = int(config.get("runner", {}).get("repeats", 1))

    jobs = []
    job_id = 0
    for model in models:
        model_id = model["id"]
        backend = model.get("backend", "hf")
        trust_remote_code = model.get("trust_remote_code", False)

        for condition in conditions:
            for task_id in tasks:
                for repeat in range(1, repeats + 1):
                    jobs.append({
                        "job_id": job_id,
                        "model_id": model_id,
                        "backend": backend,
                        "trust_remote_code": trust_remote_code,
                        "condition": condition,
                        "task": task_id,
                        "repeat": repeat
                    })
                    job_id += 1

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)

    print(f"Generated {len(jobs)} jobs to {output_path}")
    print(f"Job array indices: 0-{len(jobs)-1}")


if __name__ == "__main__":
    main()
