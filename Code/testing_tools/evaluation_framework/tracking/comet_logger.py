import csv
import io
import os
from typing import Dict, Iterable, List, Optional

from comet_ml import API, Experiment


class CometLogger:
    def __init__(self, workspace: str, project: str):
        comet_key = os.getenv("COMET_API_KEY")
        if not comet_key:
            raise ValueError("Missing COMET_API_KEY in environment or .env")
        self._api_key = comet_key
        self.workspace = workspace
        self.project = project
        self.experiment = Experiment(
            api_key=comet_key,
            workspace=workspace,
            project_name=project,
        )

    def log_params(self, params: Dict) -> None:
        self.experiment.log_parameters(params)

    def log_metric(self, name: str, value: float) -> None:
        self.experiment.log_metric(name, value)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        self.experiment.log_metrics(metrics)

    def log_table_from_rows(self, table_name: str, rows: Iterable, headers: Iterable[str]) -> None:
        self.experiment.log_table(table_name, rows, headers=headers)

    def download_table(
        self,
        experiment_key: str,
        asset_name: str = "qa_library.csv",
    ) -> List[Dict]:
        api = API(api_key=self._api_key)
        experiment = api.get_experiment(
            workspace=self.workspace,
            project_name=self.project,
            experiment=experiment_key,
        )
        asset_list = experiment.get_asset_list()
        target = next((a for a in asset_list if a["fileName"] == asset_name), None)
        if target is None:
            raise FileNotFoundError(
                f"Asset '{asset_name}' not found in experiment {experiment_key}"
            )
        raw = experiment.get_asset(target["assetId"])
        text = raw if isinstance(raw, str) else raw.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        return list(reader)

    def end(self) -> None:
        self.experiment.end()
