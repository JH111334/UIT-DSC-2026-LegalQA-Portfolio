from typing import List

def create_answer_only_labels(input_ids: List[int], prompt_token_length: int, ignore_index: int = -100) -> List[int]:
    """
    Constructs target label sequence for instruction fine-tuning.
    Tokens belonging to the instruction and context prompt are masked with ignore_index (-100),
    so loss is strictly computed over generated answer tokens.
    """
    labels = list(input_ids)
    for i in range(min(prompt_token_length, len(labels))):
        labels[i] = ignore_index
    return labels
