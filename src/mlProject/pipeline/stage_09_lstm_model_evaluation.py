from mlProject.config.configuration import ConfigurationManager
from mlProject.components.lstm_model_evaluation import LSTMModelEvaluation
from mlProject import logger


STAGE_NAME = "LSTM Model Evaluation"


class LSTMModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        deep_model_evaluation_config = config.get_deep_model_evaluation_config()
        lstm_evaluation = LSTMModelEvaluation(config=deep_model_evaluation_config)
        lstm_evaluation.log_into_mlflow()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = LSTMModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
