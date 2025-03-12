from mlProject.config.configuration import ConfigurationManager
from mlProject.components.crypto_data_ingestion import CryptoDataIngestion
from mlProject import logger


STAGE_NAME = "Crypto Data Ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        cryptocurrencies = config.get_cryptocurrencies()
        
        data_ingestion = CryptoDataIngestion(config=data_ingestion_config)
        data_ingestion.download_file(cryptocurrencies=cryptocurrencies)
        data_ingestion.extract_zip_file()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
