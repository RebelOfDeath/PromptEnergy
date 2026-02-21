import os


def _apply_condition(task_text: str) -> str:
    condition = os.getenv("PROMPT_CONDITION", "baseline_single_shot")

    if condition == "polite_single_shot":
        return f"Please help with the following task.\n\n{task_text}\n\nThanks!"
    if condition == "think_step_by_step":
        return (
            f"{task_text}\n\n"
            "Think step-by-step. First, write your reasoning. Then provide the final output."
        )
    if condition == "answer_only_no_expl":
        return f"{task_text}\n\nDo not provide explanations."

    return task_text


def humaneval_doc_to_text(doc: dict) -> str:
    base = f"{doc['prompt']}"
    return _apply_condition(base)


def humaneval_instruct_doc_to_text(doc: dict) -> str:
    base = (
        "Write a solution to the following problem and make sure that it passes the tests:\n"
        "```python\n"
        f"{doc['prompt']}\n"
        "```\n"
    )
    return _apply_condition(base)


def mbpp_doc_to_text(doc: dict) -> str:
    base = (
        "You are an expert Python programmer, and here is your task: "
        f"{doc['text']} "
        "Your code should pass these tests:\n\n"
        f"{doc['test_list'][0]}\n"
        f"{doc['test_list'][1]}\n"
        f"{doc['test_list'][2]}\n"
        "[BEGIN]\n"
    )
    return _apply_condition(base)


def code2text_doc_to_text(doc: dict) -> str:
    inputs = " ".join(doc["code_tokens"]).replace("\n", " ")
    inputs = " ".join(inputs.strip().split())
    return _apply_condition(inputs)
