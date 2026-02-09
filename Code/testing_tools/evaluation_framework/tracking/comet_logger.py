import os
from typing import Dict, Iterable

from comet_ml import Experiment


class CometLogger:
    def __init__(self, workspace: str, project: str):
        comet_key = os.getenv("COMET_API_KEY")
        if not comet_key:
            raise ValueError("Missing COMET_API_KEY in environment or .env")
        self.experiment = Experiment(
            api_key=comet_key,
            workspace=workspace,
            project_name=project,
        )

    def log_params(self, params: Dict) -> None:
        self.experiment.log_parameters(params)

    def log_table_from_rows(self, table_name: str, rows: Iterable, headers: Iterable[str]) -> None:
        self.experiment.log_table(table_name, rows, headers=headers)

    def end(self) -> None:
        self.experiment.end()
