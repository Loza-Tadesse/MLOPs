from mlProject.config.configuration import ConfigurationManager
from mlProject.components.deep_model_trainer import DeepModelTrainer
from mlProject import logger
from pathlib import Path


STAGE_NAME = "Deep Model Training stage"


class DeepModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Check if data validation passed
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                deep_model_trainer_config = config.get_deep_model_trainer_config()
                deep_model_trainer = DeepModelTrainer(config=deep_model_trainer_config)
                deep_model_trainer.train()
            else:
                raise Exception("Data schema validation failed. Cannot train deep model.")

        except Exception as e:
            logger.exception(e)
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DeepModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e