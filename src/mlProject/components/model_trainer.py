import pandas as pd
import os
from mlProject import logger
from sklearn.ensemble import RandomForestRegressor
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject.components.feature_validator import FeatureValidator


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]].values.ravel()
        test_y = test_data[[self.config.target_column]].values.ravel()

        # Save feature metadata for validation
        logger.info("Calculating feature statistics for validation")
        feature_stats = FeatureValidator.calculate_feature_statistics(train_x)
        feature_names = list(train_x.columns)
        
        model_config = {
            'model_type': 'RandomForest',
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'min_samples_split': self.config.min_samples_split,
            'target_column': self.config.target_column
        }
        
        FeatureValidator.save_feature_metadata(
            feature_names=feature_names,
            feature_stats=feature_stats,
            model_config=model_config,
            output_path='artifacts/feature_metadata.json'
        )
        logger.info("Saved feature metadata to artifacts/feature_metadata.json")

        model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        model.fit(train_x, train_y)

        joblib.dump(model, os.path.join(
            self.config.root_dir, self.config.model_name))
