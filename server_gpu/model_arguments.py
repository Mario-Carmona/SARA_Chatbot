
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    
    model_conver: str = field(
        default=None, 
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_conver_config: Optional[str] = field(
        default=None, 
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        }
    )
    model_conver_tokenizer: Optional[str] = field(
        default=None, 
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        }
    )
    model_conver_tokenizer_config: Optional[str] = field(
        default=None, 
        metadata={
            "help": "Pretrained tokenizer config name or path if not the same as model_name"
        }
    )
    
    # --------------------------------------------------

    model_trans_ES_EN: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_trans_ES_EN_config: Optional[str] = field(
        default=None, 
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        }
    )
    model_trans_ES_EN_tokenizer: Optional[str] = field(
        default=None, 
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        }
    )
    model_trans_ES_EN_tokenizer_config: Optional[str] = field(
        default=None, 
        metadata={
            "help": "Pretrained tokenizer config name or path if not the same as model_name"
        }
    )

    # --------------------------------------------------

    model_trans_EN_ES: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_trans_EN_ES_config: Optional[str] = field(
        default=None, 
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        }
    )
    model_trans_EN_ES_tokenizer: Optional[str] = field(
        default=None, 
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        }
    )
    model_trans_EN_ES_tokenizer_config: Optional[str] = field(
        default=None, 
        metadata={
            "help": "Pretrained tokenizer config name or path if not the same as model_name"
        }
    )
