from dataclasses import dataclass

@dataclass
class Argument:
    # --- Sampling ---
    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9

    # --- Length Control ---
    max_new_tokens: int = 128
    min_new_tokens: int = 10
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3

    # --- Beam Search ---
    early_stopping: bool = True
    num_beams: int = 4
    length_penalty: float = 1.0