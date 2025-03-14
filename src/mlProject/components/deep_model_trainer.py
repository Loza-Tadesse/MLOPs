"""PyTorch Deep Learning Model Trainer - Pre-trained models placeholder"""
from mlProject import logger
from mlProject.entity.config_entity import DeepModelTrainerConfig


class DeepModelTrainer:
    def __init__(self, config: DeepModelTrainerConfig):
        self.config = config
        logger.info("Using pre-trained PyTorch models from artifacts/")
        
    def train(self):
        logger.warning("Training skipped - using pre-trained PyTorch models")
        logger.info("Models: artifacts/deep_model_trainer/best_deep_model.pth")
