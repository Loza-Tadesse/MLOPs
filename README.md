
# 🚀 CryptoPredict AI - Cryptocurrency Price Prediction with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.2.2-green.svg)](https://mlflow.org)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey.svg)](https://flask.palletsprojects.com)
[![MLOps](https://img.shields.io/badge/MLOps-V2.1-brightgreen.svg)](./MLOPS_REFACTORING.md)

**Advanced AI-powered cryptocurrency price prediction system using both traditional machine learning and deep learning approaches with production-grade MLOps practices.**

## 🎯 Features

- 🤖 **Dual AI Models**: Random Forest ML (R²=0.993) & PyTorch Deep Learning (R²=0.839)
- 📊 **Real-time Data**: Live crypto prices via CoinGecko API
- 📈 **Technical Analysis**: 20+ professional trading indicators (RSI, MACD, Bollinger Bands)
- 🎨 **Modern Dashboard**: Interactive web interface with live charts
- 🔄 **Multi-Currency**: Bitcoin, Ethereum, Cardano, Solana, Ripple
- 📱 **Responsive Design**: Works on desktop and mobile
- 🧪 **MLOps Pipeline**: Complete ML workflow with experiment tracking
- ✅ **Feature Validation**: Automatic drift detection and quality checks
- 🔬 **Unified Components**: Shared feature engineering for training & prediction

## 🆕 What's New in V2.1

- ✅ **MLOps Refactoring**: Production-ready architecture with best practices
- ✅ **Feature Engineering Component**: Centralized, reusable feature calculation
- ✅ **Feature Validation**: Automatic drift detection and data quality checks  
- ✅ **Metadata Tracking**: Feature statistics saved and validated
- ✅ **Unified Pipeline**: Same feature engineering in training and production
- ✅ **Configuration-Driven**: All settings in YAML config files

👉 See [MLOPS_REFACTORING.md](./MLOPS_REFACTORING.md) for complete details

## 🏗️ Architecture

### V2.1 - MLOps Architecture
```
CryptoPredict/
├── 📊 Data Ingestion      → CoinGecko API + Historical Data
├── ✅ Data Validation     → Schema & Quality Checks  
├── 🔄 Transformation      → Train/Test Split
├── 🎨 Feature Engineering → Technical Indicators (NEW)
│   ├── RSI, MACD, Bollinger Bands
│   ├── Moving Averages (SMA, EMA)
│   └── Volatility & Price Position
├── 🤖 Model Training      → Random Forest + PyTorch Neural Networks
│   └── Feature Metadata → Statistics & Validation
├── ✅ Feature Validation  → Drift Detection (NEW)
├── 📈 Evaluation          → MLflow Tracking & Metrics
└── 🌐 Web Interface       → Flask + Real-time Dashboard

NEW in V2.1:
• FeatureEngineering: Shared component for training & prediction
• FeatureValidator: Automatic drift detection
• Metadata Tracking: Feature statistics for validation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Loza-Tadesse/VinoPredict.git
cd VinoPredict

# 2. Create virtual environment
conda create -n cryptopredict python=3.8 -y
conda activate cryptopredict

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train traditional ML model
python main_crypto.py

# 5. Train deep learning model (optional)
python main_crypto_deep.py

# 6. Launch web application
python app.py
```

### 🌐 Access the Application
Open your browser and go to: **http://localhost:8080**

## 💻 Usage

### 1. **Live Dashboard**
- View real-time crypto prices
- Monitor market changes
- Interactive price charts

### 2. **AI Prediction**
- Enter technical indicators
- Choose AI model (Traditional ML or Deep Learning)
- Get 24-hour price predictions

### 3. **Model Training**
- Traditional: `http://localhost:8080/train`
- Deep Learning: `http://localhost:8080/train_deep`

## 📊 Technical Indicators Used

| Indicator | Description | Type |
|-----------|-------------|------|
| **RSI** | Relative Strength Index | Momentum |
| **SMA** | Simple Moving Average (7, 14, 30 days) | Trend |
| **EMA** | Exponential Moving Average | Trend |
| **Bollinger Bands** | Volatility bands | Volatility |
| **Volume Ratio** | Trading volume analysis | Volume |
| **Price Position** | Position within recent range | Position |

## 🤖 AI Models

### 1. Traditional Machine Learning
- **Algorithm**: Random Forest Regression
- **Features**: Ensemble of decision trees with 100 estimators
- **Speed**: Fast training & prediction
- **Use Case**: Robust predictions with feature importance analysis

### 2. Deep Learning
- **Framework**: PyTorch
- **Architecture**: Multi-layer Neural Network
- **Features**: Advanced pattern recognition
- **Use Case**: Complex market analysis

## 📈 Data Pipeline

```python
# Example: Technical Indicators Calculation
def add_technical_indicators(df):
    # RSI Calculation
    df['rsi'] = calculate_rsi(df['price'], window=14)
    
    # Moving Averages
    df['sma_7'] = df['price'].rolling(window=7).mean()
    df['ema_14'] = df['price'].ewm(span=14).mean()
    
    # Bollinger Bands
    df['bb_upper'] = sma + (2 * std)
    df['bb_lower'] = sma - (2 * std)
    
    return df
```

## 🛠️ Configuration

### Crypto Settings (`params.yaml`)
```yaml
RandomForest:
  n_estimators: 100
  max_depth: 20
  min_samples_split: 5

DeepLearning:
  hidden_layers: [128, 64, 32]
  dropout_rate: 0.2
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
```

## 📊 MLflow Tracking

Monitor your experiments at: **http://localhost:5000**

```bash
mlflow ui
```

## 🎨 Screenshots

### Dashboard
![Dashboard Preview](https://via.placeholder.com/800x400?text=Live+Crypto+Dashboard)

### Prediction Results
![Results Preview](https://via.placeholder.com/800x400?text=AI+Prediction+Results)

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/predict` | POST | Make predictions |
| `/train` | GET | Train traditional model |
| `/train_deep` | GET | Train deep learning model |
| `/get_crypto_data/<id>` | GET | Live crypto prices |

## 📦 Project Structure

```
VinoPredict/
├── src/mlProject/
│   ├── components/          # ML components
│   ├── pipeline/           # Training pipelines  
│   ├── config/             # Configuration management
│   └── utils/              # Utility functions
├── templates/              # Web interface
├── static/                # CSS, JS, images
├── config/                # YAML configurations
├── artifacts/             # Model outputs
└── research/              # Jupyter notebooks
```

## 🚨 Disclaimer

**⚠️ Important**: This application is for **educational purposes only** and should **not** be used as financial advice. Cryptocurrency markets are highly volatile and unpredictable. Always do your own research before making investment decisions.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CoinGecko API](https://coingecko.com) for cryptocurrency data
- [PyTorch](https://pytorch.org) for deep learning framework
- [MLflow](https://mlflow.org) for experiment tracking
- [Chart.js](https://chartjs.org) for interactive charts

---

<div align="center">

**Built with ❤️ for the crypto community**

[⭐ Star this repo](https://github.com/Loza-Tadesse/VinoPredict) | [🐛 Report Bug](https://github.com/Loza-Tadesse/VinoPredict/issues) | [💡 Request Feature](https://github.com/Loza-Tadesse/VinoPredict/issues)

</div>
---

## 🧹 Codebase v2.0 - Cleaned & Optimized

### Recent Improvements

**✅ Files Removed (10):**
- `main.py` - Redundant training script
- `test.py` - Empty file
- `template.py` - Scaffolding script (no longer needed)
- `test_ml_pipeline.py` - Partial test script
- `test_predictions.py` - Old test script
- `src/mlProject/components/data_ingestion.py` - Legacy data ingestion component
- `src/mlProject/components/pytorch_model_trainer.py` - Duplicate
- `src/mlProject/pipeline/prediction.py` - Old prediction pipeline
- Log files and cache cleaned

**🔄 Files Unified:**
- `main_crypto.py` + `main_crypto_deep.py` → `train.py` (single training entry point)

**📉 Code Reduction:**
- `app.py`: 693 → 623 lines (10% reduction)
- Removed 70+ lines of redundant code
- Consolidated imports and constants
- Simplified functions

**🎯 Result:**
- Cleaner project structure
- Single training command
- Better maintainability
- Faster execution
- Professional codebase

### Training Command (New)

```bash
# Train all models (recommended)
python train.py

# Options:
--model [traditional|deep|both]  # Choose model type (default: both)
--skip-data                      # Skip data ingestion (use existing)

# Examples:
python train.py --model deep     # Train only deep learning model
python train.py --skip-data      # Fast re-training
```

### Core Files (After Cleanup)

```
Root Level (3 Python files):
  ├── app.py      (623 lines) - Flask web server
  ├── train.py    (105 lines) - Unified training
  └── setup.py    - Package setup

Components (8 files):
  ├── crypto_data_ingestion.py
  ├── data_transformation.py
  ├── data_validation.py
  ├── deep_model_evaluation.py
  ├── deep_model_trainer.py
  ├── model_evaluation.py
  └── model_trainer.py

Pipelines (9 files):
  ├── stage_01_data_ingestion.py
  ├── stage_02_data_validation.py
  ├── stage_03_data_transformation.py
  ├── stage_04_model_trainer.py
  ├── stage_05_model_evaluation.py
  ├── stage_06_deep_model_trainer.py
  ├── stage_07_deep_model_evaluation.py
  └── unified_prediction.py
```

### Performance Metrics (Maintained)

- **Prediction Accuracy**: ~0.3% average error
- **Traditional Model**: R² = 0.993, RMSE = 1.67
- **Deep Learning**: R² = 0.839, RMSE = 8.17
- **Ensemble Confidence**: 0.85
- **Training Samples**: 1,443

