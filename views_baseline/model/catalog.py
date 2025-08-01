from views_baseline.model.baseline import ZeroModel
from views_baseline.model.baseline import LocfModel


class BaselineModelCatalog:
    def __init__(self, config: dict):
        self.config = config
        self.models = {
            "ZeroModel": self._get_zero_model,
            "LocfModel": self._get_locf_model,
        }

    def get_model(self, model_name: str):
        return self.models[model_name]()

    def list_models(self):
        return list(self.models.keys())

    def _get_zero_model(self):
        return ZeroModel(targets=self.config["targets"])
    
    def _get_locf_model(self):
        return LocfModel(targets=self.config["targets"])


