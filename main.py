"""Kserve inference script."""
import re
import argparse
from typing import List
import os
import requests
import torch
from joeynmt.prediction import predict, prepare
from joeynmt.config import load_config, parse_global_args
from kserve.utils.utils import generate_uuid
from kserve import (Model, ModelServer, model_server, InferRequest, InferOutput, InferResponse)

# Set environment variables for PyTorch optimization
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
CLEANING_REGEX = re.compile(r'[!"&\(\),-./:;=?+.\[\]«»]')

def clean_text(text: str) -> str:
    """
    Clean input text by removing special characters and converting
    to lower case.
    """
    text = text.lower()
    text = CLEANING_REGEX.sub(' ', text)
    return text.strip()

class JoeyNMTModelDyuFr:
    """
    JoeyNMTModelDyuFr which loads JoeyNMT model for inference.

    :param config_path: Path to YAML config file
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, config_path: str,beam_size:int=1, n_best: int = 1) -> None:
        seed = 42
        torch.manual_seed(seed)
        cfg = load_config(config_path)
        args = parse_global_args(cfg, rank=0, mode="translate")
        self.args = args._replace(test=args.test._replace(n_best=n_best,
                                                          beam_size=beam_size,
                                                          generate_unk=False))
        self.model, _, _, self.test_data = prepare(self.args, rank=0, mode="translate")

    def _translate_data(self) -> List[str]:
        _, _, hypotheses, _, _, _ = predict(
            model=self.model,
            data=self.test_data,
            compute_loss=False,
            device=self.args.device,
            rank=0,
            n_gpu=self.args.n_gpu,
            normalization="none",
            num_workers=self.args.num_workers,
            args=self.args.test,
            autocast=self.args.autocast,
        )
        return hypotheses

    def translate(self, sentence: str) -> str:
        """
        Translate the given sentence.
        :param sentence: Sentence to be translated
        :return: The most probable translation of the sentence.
        """
        self.test_data.set_item(sentence.strip())
        translations = self._translate_data()
        assert len(translations) == len(self.test_data) * self.args.test.n_best
        self.test_data.reset_cache()
        return translations[0]

class MyModel(Model):
    """
    MyModel class for KServe inference.
    """
    def __init__(self, name: str, config_path: str):
        super().__init__(name)
        self.name = name
        self.config_path = config_path
        self.model = None
        self.ready = False
        self.load()

    def load(self):
        self.model = JoeyNMTModelDyuFr(config_path=self.config_path, n_best=1)
        self.ready = True

    async def preprocess(self, payload: InferRequest, *_args, **_kwargs) -> List[str]:
        infer_inputs: List[str] = payload.inputs[0].data
        cleaned_texts: List[str] = [clean_text(i) for i in infer_inputs]
        return cleaned_texts

    # pylint: disable=arguments-differ
    async def predict(self, data: List[str], *_predict_args, **_kwargs) -> InferResponse:
        response_id = generate_uuid()
        results = [self.model.translate(sentence=s) for s in data]
        infer_output = InferOutput(
            name="output-0", shape=[len(results)], datatype="STR", data=results
        )
        infer_response = InferResponse(
            model_name=self.name, infer_outputs=[infer_output], response_id=response_id
        )
        return infer_response

def warm_up_model(model_server_address: str):
    """
    Warm-up the model by sending a dummy request.
    """
    dummy_data = ["This is a warm-up request."]
    try:
        response = requests.post(f"http://{model_server_address}",
                                 json={"inputs": [{"data": dummy_data}]}, timeout=5)
        if response.status_code == 200:
            print("Model warm-up successful.")
        else:
            print("Model warm-up failed with status code:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("Model warm-up failed due to request exception:", e)

parser = argparse.ArgumentParser(parents=[model_server.parser], conflict_handler='resolve')
parser.add_argument(
    "--model_name",
    default="model",
    help="The name that the model is served under."
)
parser.add_argument(
    "--config_path",
    default="/app/model_dir/config.yaml",
    help="Path to the YAML config file."
)

parser.add_argument(
    "--model_server_address",
    default="localhost:8080",
    help="Address of the model server."
)

parsed_args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = MyModel(parsed_args.model_name, parsed_args.config_path)
    ModelServer().start([model])
    warm_up_model(parsed_args.model_server_address)
