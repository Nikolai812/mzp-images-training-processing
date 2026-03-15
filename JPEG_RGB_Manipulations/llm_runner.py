from abc import ABC, abstractmethod

class LLMRunner(ABC):
    @abstractmethod
    def train(self, epochs, model_path):
        pass

    @abstractmethod
    def predict(self, model_path):
        pass
