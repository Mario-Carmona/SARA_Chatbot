
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceArguments:
    do_inference: bool = field(
        metadata={
            "help": ""
        }
    )
    replace_method: str = field(
        metadata={
            "help": ""
        }
    )
    replace_with_kernel_inject: bool = field(
        metadata={
            "help": ""
        }
    )
    quantize_groups: Optional[int] = field(
        metadata={
            "help": ""
        }
    )
    mlp_exra_grouping: Optional[bool] = field(
        metadata={
            "help": ""
        }
    )
