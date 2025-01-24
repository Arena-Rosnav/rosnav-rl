from pydantic import BaseModel, Field, model_validator
from typing import Union, List
from pathlib import Path


class TransferWeightsCfg(BaseModel):
    source_dir: Union[str, Path] = Field(
        ..., description="Directory where the source agent's model is saved."
    )
    source_checkpoint: str = Field(
        "last_model", description="Checkpoint to load from the source agent."
    )
    include: Union[List[str], None] = Field(
        default_factory=list,
        description="List of regex patterns for keys to include in the transfer. If None, no weights will be transferred.",
    )
    exclude: Union[List[str], None] = Field(
        ["---"],
        description="List of substrings for keys to exclude from the transfer.",
    )

    @model_validator(mode="after")
    def check_source_dir(self):
        if not Path(self.source_dir).exists():
            raise ValueError(f"Source directory {self.source_dir} does not exist.")
        self.source_dir = Path(self.source_dir)
        return self
