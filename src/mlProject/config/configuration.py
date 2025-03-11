from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import (DataIngestionConfig)
from mlProject.entity.config_entity import (DataValidationConfig)
from mlProject.entity.config_entity import (DataTransformationConfig)
from mlProject.entity.config_entity import (ModelTrainerConfig)
from mlProject.entity.config_entity import (ModelEvaluationConfig)
from mlProject.entity.config_entity import (DeepModelTrainerConfig)
from mlProject.entity.config_entity import (DeepModelEvaluationConfig)
from mlProject.entity.config_entity import (FeatureEngineeringConfig)


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH,
            schema_filepath=SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_cryptocurrencies(self) -> list:
        """Get list of cryptocurrencies from configuration"""
        return self.config.cryptocurrencies

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.RandomForest
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            target_column=schema.name

        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.RandomForest
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            all_params=params,
            metric_file_name=config.metric_file_name,
            target_column=schema.name,
            mlflow_uri="https://dagshub.com/Loza-Tadesse/MLOPs.mlflow",

        )

        return model_evaluation_config

    def get_deep_model_trainer_config(self) -> DeepModelTrainerConfig:
        config = self.config.deep_model_trainer
        params = self.params.DeepLearning
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        deep_model_trainer_config = DeepModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            hidden_layers=params.hidden_layers,
            dropout_rate=params.dropout_rate,
            learning_rate=params.learning_rate,
            batch_size=params.batch_size,
            epochs=params.epochs,
            early_stopping_patience=params.early_stopping_patience,
            target_column=schema.name
        )

        return deep_model_trainer_config

    def get_deep_model_evaluation_config(self) -> DeepModelEvaluationConfig:
        config = self.config.deep_model_evaluation
        params = self.params.DeepLearning
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        deep_model_evaluation_config = DeepModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            scaler_path=config.scaler_path,
            model_config_path=config.model_config_path,
            all_params=params,
            metric_file_name=config.metric_file_name,
            target_column=schema.name,
            mlflow_uri="https://dagshub.com/Loza-Tadesse/MLOPs.mlflow",
        )

        return deep_model_evaluation_config

    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        config = self.config.feature_engineering

        create_directories([config.root_dir])

        feature_engineering_config = FeatureEngineeringConfig(
            root_dir=config.root_dir,
            metadata_path=config.metadata_path,
            enable_validation=config.enable_validation,
            drift_threshold=config.drift_threshold
        )

        return feature_engineering_config
