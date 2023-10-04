from transformers import HfArgumentParser
from typing import Literal, Optional
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class ModelArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model."}
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Adopt scaled rotary positional embeddings."}
    )

    def __post_init__(self):
        if self.quantization_bit is not None:
            assert self.quantization_bit in [4, 8], "We only accept 4-bit or 8-bit quantization."


@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    dataset: Optional[str] = field(
        default="alpaca_en",
        metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."}
    )
    dataset_dir: Optional[str] = field(
        default="data",
        metadata={"help": "The name of the folder containing datasets."}
    )
    split: Optional[str] = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."}
    )


def parse_args() -> Tuple[ModelArguments, DataArguments]:
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
    ))
    return parser.parse_args_into_dataclasses()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parse_args()
    print(model_args, data_args)
