from dataclasses import dataclass


@dataclass
class InputArguments:
    text:bool = False
    image:bool = False
    video:bool = False
    audio:bool = False
    file:bool = False
    examples: str|list[str]|None = None

@dataclass
class HyperParameterArguments:
    seed:int|None = 42
    do_sample:bool|None = None
    temperature:float|None = None
    top_k:int|None = None
    top_p:float|None = None
    min_new_tokens:int|None = None
    max_new_tokens:int|None = None
    repetition_penalty:float|None = None
    no_repeat_ngram_size:int|None = None
    do_beam_search:bool|None = None
    early_stopping:bool|None = None
    num_beams:int|None = None
    length_penalty:float|None = None


@dataclass
class OutputArguments:
    text:bool = False
    image:bool = False
    video:bool = False
    audio:bool = False
    file:bool = False
