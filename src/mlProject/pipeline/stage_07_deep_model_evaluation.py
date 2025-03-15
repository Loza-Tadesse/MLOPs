from mlProject.config.configuration import ConfigurationManager
from mlProject.components.deep_model_evaluation import DeepModelEvaluation
from mlProject import logger


STAGE_NAME = "Deep Model Evaluation stage"


class DeepModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            deep_model_evaluation_config = config.get_deep_model_evaluation_config()
            deep_model_evaluation = DeepModelEvaluation(config=deep_model_evaluation_config)
            deep_model_evaluation.log_into_mlflow()

        except Exception as e:
            logger.exception(e)
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DeepModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e