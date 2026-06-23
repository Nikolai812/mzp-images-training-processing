from abc import ABC, abstractmethod
import ast

class LLMRunner(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, model_path):
        pass

    @abstractmethod
    def load_model(self, model_filename):
        pass

    def __init__(self, runner_config):
        self.runner_config = runner_config
        self.root_dir = runner_config['training_input']
        self.raw_input_dir = runner_config['raw_input']
        self.models_dir = runner_config['models']
        self.output_json = runner_config['output_json']

        self.classes = ast.literal_eval(runner_config['training_categories'])