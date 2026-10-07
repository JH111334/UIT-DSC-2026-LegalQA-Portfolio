from src.generation.masking import create_answer_only_labels

def test_create_answer_only_labels():
    input_ids = [101, 2054, 2003, 1037, 102, 3405, 102]
    prompt_len = 5

    labels = create_answer_only_labels(input_ids, prompt_token_length=prompt_len, ignore_index=-100)
    assert labels[:5] == [-100, -100, -100, -100, -100]
    assert labels[5:] == [3405, 102]
