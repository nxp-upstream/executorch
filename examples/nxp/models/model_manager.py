# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging
import os
from enum import Enum

import torch
import yaml

import executorch.examples.nxp.models.utils as utils

logging.basicConfig(level=logging.INFO)


class ModelSource(Enum):
    MLPERF_TINY = 0

MODEL_NAME_TO_MODEL_SOURCE = {
    "visual_wake_words": ModelSource.MLPERF_TINY,
    "anomaly_detection": ModelSource.MLPERF_TINY,
    "keyword_spotting": ModelSource.MLPERF_TINY,
    "image_classification": ModelSource.MLPERF_TINY
}

class ModelManager:
    # TODO: Update data_path if models will be moved to executorch/models.
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data")
    mlperf_tiny_path = os.path.join(data_path, "mlperf_tiny")
    mlperf_tiny_py_path = "executorch.data.mlperf_tiny."
    models_path = "benchmark/experimental/training_torch"
    models_py_path = "benchmark.experimental.training_torch."

    def _get_mlperf_tiny_model(self, model_name: str, model_config: dict, **kwargs) -> torch.nn.Module:
        logging.info(f"Loading MLPerf Tiny model {model_name}...")
        model = self.module.model(**kwargs)
        model.eval()
        logging.info("Model loaded successfully.")

        if "model_weights_path" in model_config and model_config["model_weights_path"] is not None:
            mlperf_tiny_model_weights_path = os.path.join(self.mlperf_tiny_path, self.models_path, model_name,
                                                          model_config["model_weights_path"])
            model.load_state_dict(torch.load(mlperf_tiny_model_weights_path, weights_only=True))
            logging.info("Model weights loaded successfully.")

        return model

    def _get_mlperf_tiny_resource(self, model_name: str, resource: str, **kwargs) -> torch.nn.Module:
        function_resources = {
            "model": self._get_mlperf_tiny_model,
        }
        self._load_mlperf_tiny_repo(model_name)
        mlperf_tiny_model_config_path = os.path.join(self.mlperf_tiny_path, self.models_path, model_name, "config.yaml")
        model_config = yaml.safe_load(open(mlperf_tiny_model_config_path))

        if resource not in function_resources:
            raise ValueError(f"Model {model_name} has no resource {resource}!")
        return function_resources[resource](model_name, model_config, **kwargs)

    def _load_mlperf_tiny_repo(self, model_name: str):
        mlperf_tiny_url = "ssh://git@bitbucket.sw.nxp.com/aitec/mlperf_tiny.git"
        mlperf_tiny_model_py_path = self.mlperf_tiny_py_path + self.models_py_path + model_name

        os.makedirs(self.data_path, exist_ok=True)

        try:
            self.module = importlib.import_module(mlperf_tiny_model_py_path)
        except ModuleNotFoundError:
            # TODO: As the MLPerf tiny models are not yet publicly available (i.e. only on NXP's BitBucket) we clone
            #  the whole repository. Alternatively we could use the direct https url to the file but that required,
            #  to log in with credentials. The git uses ssh key for authentication.
            if os.path.isdir(os.path.join(self.mlperf_tiny_path, ".github")):
                utils.run_cmd("git pull", "Failed to pull from MLPerf Tiny repository.",
                              logging, self.mlperf_tiny_path)
                utils.run_cmd("git reset --hard", "", logging, self.mlperf_tiny_path)
                logging.info("Pulled from MLPerf Tiny repository.")
            else:
                utils.run_cmd(f"git clone {mlperf_tiny_url}", "Failed to clone the MLPerf Tiny repository.",
                              logging, self.data_path)
                logging.info("Successfully cloned the MLPerf Tiny repository.")
            try:
                self.module = importlib.import_module(mlperf_tiny_model_py_path)
            except ImportError:
                raise ValueError(f"Model {model_name} not found in MLPerf Tiny repository!")

    def _select_source(self, model_name: str, resource: str, **kwargs) -> torch.nn.Module:
        if model_name not in MODEL_NAME_TO_MODEL_SOURCE:
            raise ValueError(f"Model {model_name} not supported!")

        model_source = MODEL_NAME_TO_MODEL_SOURCE[model_name]
        if model_source == ModelSource.MLPERF_TINY:
            return self._get_mlperf_tiny_resource(model_name, resource, **kwargs)
        else:
            raise ValueError(f"Model {model_source} not supported!")

    def get_model(self, model_name: str, **kwargs) -> torch.nn.Module:
        """ Get PyTorch nn.Module model.

        :param model_name: Name of selected model.
        :param kwargs: Keyword arguments needed for model initialization.
        :return nn.Module: An instance of a PyTorch model, suitable for eager execution.
        """
        return self._select_source(model_name, "model", **kwargs)
