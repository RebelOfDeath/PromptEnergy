#!/usr/bin/env python3
"""
Custom HumanEval evaluation pipeline with edit distance, CodeBLEU, and ROUGE-L metrics.
Does not use lm-evaluation-harness - loads HumanEval directly from datasets.
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
import Levenshtein
from transformers import StoppingCriteria, StoppingCriteriaList

class StopWordCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_words):
        self.tokenizer = tokenizer
        self.stop_words = stop_words

    def __call__(self, input_ids, scores, **kwargs):
        # Decode only the last 20 tokens to save compute
        tail_text = self.tokenizer.decode(input_ids[0][-20:], skip_special_tokens=True)
        for stop_word in self.stop_words:
            if stop_word in tail_text:
                return True
        return False

def safe_name(value: str) -> str:
    """Convert string to safe filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_energy_joules(stdout: str) -> float | None:
    """Extract energy consumption from EnergiBridge output."""
    match = re.search(r"Energy consumption in joules:\s*([0-9.]+)", stdout)
    if not match:
        return None
    return float(match.group(1))


def apply_prompt_condition(text: str, condition: str) -> str:
    """Apply prompt condition formatting."""
    if condition == "polite_single_shot":
        return f"Please help with writing the following function.\n\n{text}\n\nGive your answer strictly inside a ```python\n``` code block, Thanks!"

    if condition == "think_step_by_step":
        return (
            f"{text}\n\n"
            "Think step-by-step. First, write your reasoning. Then, provide your final output strictly inside a ```python\n``` code block."
        )

    if condition == "answer_only_no_expl":
        return f"Do not provide explanations, complete the following function.\n\n{text}\n\nOutput ONLY the code strictly inside a ```python\n``` code block."

    return f"{text}\n\nPlease output your solution strictly inside a ```python\n``` code block."


def extract_code_from_response(response: str, prompt: str) -> str:
    """
    Extract code from model response.
    Handles various formats: markdown code blocks, raw code, etc.
    """
    # Remove the original prompt if it appears at the start
    if response.startswith(prompt):
        response = response[len(prompt):]

    # Try to extract from markdown code block
    code_block_pattern = r"```(?:python)?\s*\n?(.*?)```"
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        # Return the first code block
        return matches[0].strip()

    # Try to find function definition
    func_pattern = r"(def\s+\w+\s*\([^)]*\).*?)(?=\ndef\s+|\nclass\s+|\Z)"
    matches = re.findall(func_pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # If no clear code block, return everything up to common stop sequences
    stop_sequences = ["\n\n\n", "\nclass ", "\n#", "\nif __name__"]
    result = response
    for stop in stop_sequences:
        if stop in result:
            result = result[:result.index(stop)]

    return result.strip()


def compute_edit_distance(prediction: str, reference: str) -> dict[str, float]:
    """
    Compute edit distance metrics between prediction and reference.
    Returns normalized edit distance (0 = identical, 1 = completely different).
    """
    distance = Levenshtein.distance(prediction, reference)
    max_len = max(len(prediction), len(reference), 1)
    normalized = distance / max_len

    # Also compute ratio (similarity)
    ratio = Levenshtein.ratio(prediction, reference)

    return {
        "edit_distance": distance,
        "edit_distance_normalized": normalized,
        "levenshtein_ratio": ratio,
    }


def compute_rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-L score between prediction and reference."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = scorer.score(reference, prediction)

    return {
        "rouge_l_precision": scores['rougeL'].precision,
        "rouge_l_recall": scores['rougeL'].recall,
        "rouge_l_fmeasure": scores['rougeL'].fmeasure,
    }


def compute_codebleu(prediction: str, reference: str) -> dict[str, float]:
    """
    Compute CodeBLEU-like metrics.
    This is a simplified version focusing on token-level and n-gram matching.
    For full CodeBLEU, you'd need AST parsing which requires additional dependencies.
    """
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu(
            references=[[reference]],
            predictions=[prediction],
            lang="python",
            weights=(0.25, 0.25, 0.25, 0.25),
            tokenizer=None
        )
        return {
            "codebleu": result["codebleu"],
            "codebleu_ngram_match": result.get("ngram_match_score", 0.0),
            "codebleu_weighted_ngram": result.get("weighted_ngram_match_score", 0.0),
            "codebleu_syntax_match": result.get("syntax_match_score", 0.0),
            "codebleu_dataflow_match": result.get("dataflow_match_score", 0.0),
        }
    except ImportError:
        # Fallback: compute simple token-based BLEU
        from collections import Counter
        import math

        def get_ngrams(tokens: list[str], n: int) -> Counter:
            return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

        pred_tokens = prediction.split()
        ref_tokens = reference.split()

        if not pred_tokens or not ref_tokens:
            return {"codebleu": 0.0, "codebleu_ngram_match": 0.0}

        # Modified precision for n-grams 1-4
        precisions = []
        for n in range(1, 5):
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)

            if not pred_ngrams:
                precisions.append(0.0)
                continue

            matches = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
            total = sum(pred_ngrams.values())
            precisions.append(matches / total if total > 0 else 0.0)

        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens))) if pred_tokens else 0.0

        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / 4)
        else:
            geo_mean = 0.0

        bleu = bp * geo_mean

        return {
            "codebleu": bleu,
            "codebleu_ngram_match": precisions[0] if precisions else 0.0,
        }


def compute_all_metrics(prediction: str, reference: str) -> dict[str, float]:
    """Compute all metrics for a single prediction-reference pair."""
    metrics = {}

    # Edit distance metrics
    metrics.update(compute_edit_distance(prediction, reference))

    # ROUGE-L metrics
    metrics.update(compute_rouge_l(prediction, reference))

    # CodeBLEU metrics
    metrics.update(compute_codebleu(prediction, reference))

    return metrics


def load_humaneval() -> list[dict]:
    """Load HumanEval dataset from Hugging Face."""
    dataset = load_dataset("openai/openai_humaneval", split="test")
    return list(dataset)


def create_prompt(doc: dict, condition: str) -> str:
    """Create prompt from HumanEval document with condition applied."""
    base_prompt = doc["prompt"]
    return apply_prompt_condition(base_prompt, condition)


def get_canonical_solution(doc: dict) -> str:
    """Get the canonical solution for a HumanEval problem."""
    return doc["canonical_solution"]


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: str = "cuda",
) -> str:
    """Generate response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    stop_words = ["\n```\n", "\n```", "\nif __name__", "\nclass "]
    stopping_criteria = StoppingCriteriaList([StopWordCriteria(tokenizer, stop_words)])

    with torch.no_grad():
        if temperature == 0.0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def run_evaluation_with_energy(
    cmd: list[str],
    env: dict,
    energi_bridge_bin: Path,
    energy_csv: Path,
    command_output: Path,
    max_execution: int | None = None,
) -> tuple[subprocess.CompletedProcess, float | None]:
    """Run evaluation command with energy measurement."""
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
    energibridge_cmd.extend(cmd)

    proc = subprocess.run(
        energibridge_cmd,
        env=env,
        # capture_output=True,
        text=True,
    )

    energy_j = parse_energy_joules(proc.stdout)
    return proc, energy_j


def run_evaluation_direct(
    model_id: str,
    condition: str,
    output_dir: Path,
    device: str = "cuda",
    seed: int = 42,
    max_new_tokens: int = 512,
    limit: int | None = None,
    trust_remote_code: bool = False,
) -> dict:
    """
    Run HumanEval evaluation directly without energy measurement.
    Returns results dictionary.
    """
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load model and tokenizer
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=trust_remote_code,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading HumanEval dataset...")
    dataset = load_humaneval()
    if limit:
        dataset = dataset[:limit]

    print(f"Evaluating {len(dataset)} examples with condition: {condition}")

    results = []
    aggregated_metrics = {
        "edit_distance": [],
        "edit_distance_normalized": [],
        "levenshtein_ratio": [],
        "rouge_l_precision": [],
        "rouge_l_recall": [],
        "rouge_l_fmeasure": [],
        "codebleu": [],
    }

    for i, doc in enumerate(dataset):
        task_id = doc["task_id"]
        print(f"  [{i+1}/{len(dataset)}] {task_id}")

        # Create prompt
        prompt = create_prompt(doc, condition)

        # Generate response
        response = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        # Extract code from response
        extracted_code = extract_code_from_response(response, prompt)

        # Get reference
        reference = get_canonical_solution(doc)

        # Compute metrics
        metrics = compute_all_metrics(extracted_code, reference)

        # Store result
        result = {
            "task_id": task_id,
            "prompt": prompt,
            "raw_response": response,
            "extracted_code": extracted_code,
            "reference": reference,
            "metrics": metrics,
        }
        results.append(result)

        # Aggregate metrics
        for key in aggregated_metrics:
            if key in metrics:
                aggregated_metrics[key].append(metrics[key])

    # Compute mean metrics
    mean_metrics = {}
    for key, values in aggregated_metrics.items():
        if values:
            mean_metrics[f"mean_{key}"] = sum(values) / len(values)

    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "detailed_results.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save summary
    summary = {
        "model_id": model_id,
        "condition": condition,
        "num_examples": len(dataset),
        "metrics": mean_metrics,
    }
    summary_file = output_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"Mean metrics:")
    for key, value in mean_metrics.items():
        print(f"  {key}: {value:.4f}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run custom HumanEval evaluation with edit distance, CodeBLEU, and ROUGE-L"
    )
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--job-index", type=int, default=None,
                        help="Run specific job from jobs.json")
    parser.add_argument("--jobs-file", default="jobs.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-energy", action="store_true",
                        help="Run without energy measurement")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))

    project = config.get("project", {})
    output_dir = Path(args.output or project.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    runner_cfg = config.get("runner", {})
    energi_bridge_path = Path(runner_cfg.get("energi_bridge_path", "EnergiBridge"))
    device = runner_cfg.get("device", "cuda")
    if isinstance(device, str) and device.lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = runner_cfg.get("seed", 42)
    repeats = int(runner_cfg.get("repeats", 1))

    energy_cfg = config.get("energy", {})
    energy_jsonl = Path(energy_cfg.get("output_jsonl", output_dir / "energy_runs.jsonl"))
    energy_jsonl.parent.mkdir(parents=True, exist_ok=True)

    energi_bridge_bin = Path(
        os.environ.get(
            "ENERGI_BRIDGE_BIN",
            energi_bridge_path / "target" / "release" / "energibridge.exe",
        )
    )

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_root = output_dir / f"run_{run_id}"
    run_root.mkdir(parents=True, exist_ok=True)

    # If job index specified, run single job
    if args.job_index is not None:
        with open(args.jobs_file, "r") as f:
            jobs = json.load(f)

        if args.job_index >= len(jobs):
            raise ValueError(f"Job index {args.job_index} out of range")

        job = jobs[args.job_index]
        model_id = job["model_id"]
        condition = job["condition"]
        repeat = job["repeat"]
        trust_remote_code = bool(job.get("trust_remote_code", False))

        run_dir = run_root / safe_name(model_id) / condition / "humaneval_custom" / f"r{repeat}"

        print(f"Running job {args.job_index}: {model_id} | {condition} | repeat {repeat}")

        start_time = time.time()
        summary = run_evaluation_direct(
            model_id=model_id,
            condition=condition,
            output_dir=run_dir,
            device=device,
            seed=seed,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            trust_remote_code=trust_remote_code,
        )
        end_time = time.time()

        # Log to energy jsonl
        record = {
            "run_id": run_id,
            "job_index": args.job_index,
            "model_id": model_id,
            "task_id": "humaneval_custom",
            "condition_id": condition,
            "repeat": repeat,
            "duration_s": round(end_time - start_time, 3),
            "metrics": summary["metrics"],
            "output_dir": str(run_dir),
        }

        with energy_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return 0

    # Otherwise, run all combinations from config
    models = config.get("models", [])
    conditions = [c["id"] for c in config.get("prompt_conditions", [])]

    for model in models:
        model_id = model["id"]
        trust_remote_code = bool(model.get("trust_remote_code", False))

        for condition in conditions:
            for repeat in range(1, repeats + 1):
                run_dir = run_root / safe_name(model_id) / condition / "humaneval_custom" / f"r{repeat}"

                print(f"\n{'='*60}")
                print(f"Model: {model_id}")
                print(f"Condition: {condition}")
                print(f"Repeat: {repeat}/{repeats}")
                print(f"{'='*60}")

                start_time = time.time()
                summary = run_evaluation_direct(
                    model_id=model_id,
                    condition=condition,
                    output_dir=run_dir,
                    device=device,
                    seed=seed + repeat,  # Vary seed per repeat
                    max_new_tokens=args.max_new_tokens,
                    limit=args.limit,
                    trust_remote_code=trust_remote_code,
                )
                end_time = time.time()

                # Log to energy jsonl
                record = {
                    "run_id": run_id,
                    "model_id": model_id,
                    "task_id": "humaneval_custom",
                    "condition_id": condition,
                    "repeat": repeat,
                    "duration_s": round(end_time - start_time, 3),
                    "metrics": summary["metrics"],
                    "output_dir": str(run_dir),
                }

                with energy_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

    print(f"\nAll evaluations complete. Results in {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
