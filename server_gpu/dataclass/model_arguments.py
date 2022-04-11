
import os
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """
    Argumentos relacionados con los modelos usados en el proyecto
    """
    
    model_conver: str = field(
        default=None,
        metadata={
            "help": "Ruta a la carpeta del modelo"
        }
    )
    model_conver_config: str = field(
        default=None,
        metadata={
            "help": "Ruta al archivo de configuración del modelo"
        }
    )
    model_conver_tokenizer: str = field(
        default=None,
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo"
        }
    )
    model_conver_tokenizer_config: str = field(
        default=None,
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo"
        }
    )
    
    # --------------------------------------------------

    model_trans_ES_EN: str = field(
        default=None,
        metadata={
            "help": "Ruta a la carpeta del modelo"
        }
    )
    model_trans_ES_EN_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del modelo"
        }
    )
    model_trans_ES_EN_tokenizer: str = field(
        default=None, 
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo"
        }
    )
    model_trans_ES_EN_tokenizer_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo"
        }
    )

    # --------------------------------------------------

    model_trans_EN_ES: str = field(
        default=None,
        metadata={
            "help": "Ruta a la carpeta del modelo"
        }
    )
    model_trans_EN_ES_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del modelo"
        }
    )
    model_trans_EN_ES_tokenizer: str = field(
        default=None, 
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo"
        }
    )
    model_trans_EN_ES_tokenizer_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo"
        }
    )

    # --------------------------------------------------

    model_summary: str = field(
        default=None,
        metadata={
            "help": "Ruta a la carpeta del modelo"
        }
    )
    model_summary_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del modelo"
        }
    )
    model_summary_tokenizer: str = field(
        default=None, 
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo"
        }
    )
    model_summary_tokenizer_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo"
        }
    )

    # --------------------------------------------------

    model_genQuestion: str = field(
        default=None,
        metadata={
            "help": "Ruta a la carpeta del modelo"
        }
    )
    model_genQuestion_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del modelo"
        }
    )
    model_genQuestion_tokenizer: str = field(
        default=None, 
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo"
        }
    )
    model_genQuestion_tokenizer_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo"
        }
    )

    # --------------------------------------------------

    model_simplify: str = field(
        default=None,
        metadata={
            "help": "Ruta a la carpeta del modelo"
        }
    )
    model_simplify_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del modelo"
        }
    )
    model_simplify_tokenizer: str = field(
        default=None, 
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo"
        }
    )
    model_simplify_tokenizer_config: str = field(
        default=None, 
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo"
        }
    )

    def __post_init__(self):
        assert os.path.exists(self.model_conver), "`model_conver` debe ser un directorio existente."
        assert os.path.exists(self.model_conver_config), "`model_conver_config` debe ser un archivo existente."
        assert os.path.exists(self.model_conver_tokenizer), "`model_conver_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_conver_tokenizer_config), "`model_conver_tokenizer_config` debe ser un archivo existente."

        assert os.path.exists(self.model_trans_ES_EN), "`model_trans_ES_EN` debe ser un directorio existente."
        assert os.path.exists(self.model_trans_ES_EN_config), "`model_trans_ES_EN_config` debe ser un archivo existente."
        assert os.path.exists(self.model_trans_ES_EN_tokenizer), "`model_trans_ES_EN_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_trans_ES_EN_tokenizer_config), "`model_trans_ES_EN_tokenizer_config` debe ser un archivo existente."

        assert os.path.exists(self.model_trans_EN_ES), "`model_trans_EN_ES` debe ser un directorio existente."
        assert os.path.exists(self.model_trans_EN_ES_config), "`model_trans_EN_ES_config` debe ser un archivo existente."
        assert os.path.exists(self.model_trans_EN_ES_tokenizer), "`model_trans_EN_ES_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_trans_EN_ES_tokenizer_config), "`model_trans_EN_ES_tokenizer_config` debe ser un archivo existente."

        assert os.path.exists(self.model_summary), "`model_summary` debe ser un directorio existente."
        assert os.path.exists(self.model_summary_config), "`model_summary_config` debe ser un archivo existente."
        assert os.path.exists(self.model_summary_tokenizer), "`model_summary_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_summary_tokenizer_config), "`model_summary_tokenizer_config` debe ser un archivo existente."

        assert os.path.exists(self.model_genQuestion), "`model_genQuestion` debe ser un directorio existente."
        assert os.path.exists(self.model_genQuestion_config), "`model_genQuestion_config` debe ser un archivo existente."
        assert os.path.exists(self.model_genQuestion_tokenizer), "`model_genQuestion_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_genQuestion_tokenizer_config), "`model_genQuestion_tokenizer_config` debe ser un archivo existente."

        assert os.path.exists(self.model_simplify), "`model_simplify` debe ser un directorio existente."
        assert os.path.exists(self.model_simplify_config), "`model_simplify_config` debe ser un archivo existente."
        assert os.path.exists(self.model_simplify_tokenizer), "`model_simplify_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_simplify_tokenizer_config), "`model_simplify_tokenizer_config` debe ser un archivo existente."
