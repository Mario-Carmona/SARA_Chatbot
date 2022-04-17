
import os
from dataclasses import dataclass, field
from dataclass.project_arguments import ProyectArguments


@dataclass
class ModelSumArguments(ProyectArguments):
    """
    Argumentos relacionados con el modelo usado para resumir textos
    """

    model_summary: str = field(
        metadata={
            "help": "Ruta a la carpeta del modelo"
        }
    )
    model_summary_config: str = field(
        metadata={
            "help": "Ruta al archivo de configuración del modelo"
        }
    )
    model_summary_tokenizer: str = field(
        metadata={
            "help": "Ruta a la carpeta del tokenizer del modelo"
        }
    )
    model_summary_tokenizer_config: str = field(
        metadata={
            "help": "Ruta al archivo de configuración del tokenizer del modelo"
        }
    )

    def __post_init__(self):
        self.model_summary = os.path.join(self.workdir, self.model_summary)
        self.model_summary_config = os.path.join(self.workdir, self.model_summary_config)
        self.model_summary_tokenizer = os.path.join(self.workdir, self.model_summary_tokenizer)
        self.model_summary_tokenizer_config = os.path.join(self.workdir, self.model_summary_tokenizer_config)

        assert os.path.exists(self.model_summary), "`model_summary` debe ser un directorio existente."
        assert os.path.exists(self.model_summary_config), "`model_summary_config` debe ser un archivo existente."
        assert os.path.exists(self.model_summary_tokenizer), "`model_summary_tokenizer` debe ser un directorio existente."
        assert os.path.exists(self.model_summary_tokenizer_config), "`model_summary_tokenizer_config` debe ser un archivo existente."
