#!/usr/bin/env python3
"""
LSTM Training Script for CryptoPredict
Trains LSTM model for 5-minute cryptocurrency price prediction
"""
from mlProject import logger
from mlProject.pipeline.stage_08_lstm_model_trainer import LSTMModelTrainingPipeline
from mlProject.pipeline.stage_09_lstm_model_evaluation import LSTMModelEvaluationPipeline


def main():
    logger.info("🚀 Starting LSTM Model Training for CryptoPredict")
    
    # Train LSTM model
    try:
        logger.info("🧠 Training LSTM model...")
        training_pipeline = LSTMModelTrainingPipeline()
        training_pipeline.main()
        logger.info("✅ LSTM model training completed!")
    except Exception as e:
        logger.exception(f"❌ LSTM training failed: {e}")
        raise e
    
    # Evaluate LSTM model
    try:
        logger.info("📊 Evaluating LSTM model...")
        evaluation_pipeline = LSTMModelEvaluationPipeline()
        evaluation_pipeline.main()
        logger.info("✅ LSTM model evaluation completed!")
    except Exception as e:
        logger.exception(f"❌ LSTM evaluation failed: {e}")
        raise e
    
    logger.info("🎉 LSTM model training and evaluation completed successfully!")
    logger.info("💡 Model saved to: artifacts/deep_model_trainer/best_deep_model.pth")
    logger.info("💡 Start the web app with: python app.py")


if __name__ == "__main__":
    main()
