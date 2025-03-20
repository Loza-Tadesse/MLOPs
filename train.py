#!/usr/bin/env python3
"""
Unified Training Script for CryptoPredict ML Models
Trains both traditional (Random Forest) and deep learning (PyTorch) models
"""
import argparse
from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from mlProject.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from mlProject.pipeline.stage_06_deep_model_trainer import DeepModelTrainingPipeline
from mlProject.pipeline.stage_07_deep_model_evaluation import DeepModelEvaluationPipeline


def run_data_pipeline():
    """Run data ingestion, validation, and transformation stages"""
    stages = [
        ("Crypto Data Ingestion", DataIngestionTrainingPipeline),
        ("Crypto Data Validation", DataValidationTrainingPipeline),
        ("Crypto Data Transformation", DataTransformationTrainingPipeline)
    ]
    
    for stage_name, pipeline_class in stages:
        try:
            logger.info(f">>>>>> stage {stage_name} started <<<<<<")
            pipeline = pipeline_class()
            pipeline.main()
            logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e


def train_traditional_model():
    """Train traditional Random Forest model"""
    stages = [
        ("Traditional Model Trainer", ModelTrainerTrainingPipeline),
        ("Traditional Model Evaluation", ModelEvaluationTrainingPipeline)
    ]
    
    for stage_name, pipeline_class in stages:
        try:
            logger.info(f">>>>>> stage {stage_name} started <<<<<<")
            pipeline = pipeline_class()
            pipeline.main()
            logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e


def train_deep_learning_model():
    """Train deep learning PyTorch model"""
    stages = [
        ("Deep Learning Model Trainer", DeepModelTrainingPipeline),
        ("Deep Learning Model Evaluation", DeepModelEvaluationPipeline)
    ]
    
    for stage_name, pipeline_class in stages:
        try:
            logger.info(f">>>>>> stage {stage_name} started <<<<<<")
            pipeline = pipeline_class()
            pipeline.main()
            logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e


def main():
    parser = argparse.ArgumentParser(description='Train CryptoPredict ML Models')
    parser.add_argument('--model', 
                       choices=['traditional', 'deep', 'both'], 
                       default='both',
                       help='Model type to train: traditional, deep, or both (default: both)')
    parser.add_argument('--skip-data', 
                       action='store_true',
                       help='Skip data pipeline (use existing data)')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting CryptoPredict Model Training")
    logger.info(f"Model type: {args.model}")
    
    # Run data pipeline unless skipped
    if not args.skip_data:
        logger.info("📊 Running data pipeline...")
        run_data_pipeline()
    else:
        logger.info("⏩ Skipping data pipeline (using existing data)")
    
    # Train models based on selection
    if args.model in ['traditional', 'both']:
        logger.info("🔧 Training traditional Random Forest model...")
        train_traditional_model()
        logger.info("✅ Traditional model training completed!")
    
    if args.model in ['deep', 'both']:
        logger.info("🧠 Training deep learning PyTorch model...")
        train_deep_learning_model()
        logger.info("✅ Deep learning model training completed!")
    
    logger.info("🎉 All model training completed successfully!")
    logger.info("💡 Start the web app with: python app.py")


if __name__ == "__main__":
    main()
