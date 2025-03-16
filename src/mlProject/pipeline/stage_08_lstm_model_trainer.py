from mlProject.config.configuration import ConfigurationManager
from mlProject.components.lstm_model_trainer import LSTMModelTrainer
from mlProject import logger


STAGE_NAME = "LSTM Model Training"


class LSTMModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        deep_model_trainer_config = config.get_deep_model_trainer_config()
        lstm_trainer = LSTMModelTrainer(config=deep_model_trainer_config)
        lstm_trainer.train(use_lstm=True, sequence_length=10)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = LSTMModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
