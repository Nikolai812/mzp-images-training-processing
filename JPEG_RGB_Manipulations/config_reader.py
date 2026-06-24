import configparser
import os
from enum import Enum


class Framework(Enum):
    PYTORCH = "PYTORCH"
    TENSORFLOW = "TENSORFLOW"


class ConfigReader:
    def __init__(self, config_file='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def get_framework_config(self, framework: Framework):
        section = self.config[framework.value]

        model_file = section["model_file"]

        predict_from_model_file = (
                section.get("predict_from_model_file", "").strip()
                or model_file
        )

        return {
            'training_input': section['training_input'],
            'raw_input': section['raw_input'],
            'models': section['models'],
            'model_file': model_file,
            'predict_from_model_file': predict_from_model_file,
            'epochs': section['epochs'],
            'output_json': section['output_json'],
            'training_categories': section['training_categories'],
            'data_type': section['data_type']
        }


    # def get_pytorch_config(self):
    #     return {
    #         'training_input': self.config['PYTORCH']['training_input'],
    #         'raw_input': self.config['PYTORCH']['raw_input'],
    #         'models': self.config['PYTORCH']['models'],
    #         'model_file': self.config['PYTORCH']['model_file'],
    #         'predict_from_model_file': self.config['PYTORCH'].get(
    #         'predict_from_model_file',
    #         ''),
    #         'epochs': self.config['PYTORCH']['epochs'],
    #         'output_json': self.config['PYTORCH']['output_json'],
    #         'training_categories': self.config['PYTORCH']['training_categories'],
    #         'data_type': self.config['PYTORCH']['data_type']
    #     }
    #
    # def get_tensorflow_config(self):
    #     return {
    #         'training_input': self.config['TENSORFLOW']['training_input'],
    #         'raw_input': self.config['TENSORFLOW']['raw_input'],
    #         'models': self.config['TENSORFLOW']['models'],
    #         'model_file' : self.config['TENSORFLOW']['model_file'],
    #         'predict_from_model_file': self.config['TENSORFLOW'].get(
    #         'predict_from_model_file',
    #         ''),
    #         'epochs': self.config['TENSORFLOW']['epochs'],
    #         'output_json': self.config['TENSORFLOW']['output_json'],
    #         'training_categories': self.config['TENSORFLOW']['training_categories'],
    #         'data_type': self.config['TENSORFLOW']['data_type']
    #     }

    def ensure_directories_exist(self):
        os.makedirs(self.config['PYTORCH']['training_input'], exist_ok=True)
        os.makedirs(self.config['PYTORCH']['raw_input'], exist_ok=True)
        os.makedirs(self.config['PYTORCH']['models'], exist_ok=True)
        os.makedirs(self.config['TENSORFLOW']['training_input'], exist_ok=True)
        os.makedirs(self.config['TENSORFLOW']['raw_input'], exist_ok=True)
        os.makedirs(self.config['TENSORFLOW']['models'], exist_ok=True)
