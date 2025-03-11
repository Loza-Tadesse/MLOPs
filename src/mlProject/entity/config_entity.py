from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    n_estimators: int
    max_depth: int
    min_samples_split: int
    target_column: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str


@dataclass(frozen=True)
class DeepModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    hidden_layers: list
    dropout_rate: float
    learning_rate: float
    batch_size: int
    epochs: int
    early_stopping_patience: int
    target_column: str


@dataclass(frozen=True)
class DeepModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    scaler_path: Path
    model_config_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    root_dir: Path
    metadata_path: Path
    enable_validation: bool
    drift_threshold: float
