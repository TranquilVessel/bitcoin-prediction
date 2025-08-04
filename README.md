# Bitcoin Price Prediction 📈

A deep learning project using LSTM networks to forecast Bitcoin prices based on historical time series data. Built with TensorFlow, Keras, Pandas, and NumPy.

## 🔧 Features

- Data preprocessing and normalization
- Sequence creation for time series modeling
- LSTM and Bidirectional LSTM architectures
- Model training and evaluation
- Future price prediction with aligned timestamps
- Visualization of predictions vs actual prices

## 📊 Technologies

- Python
- TensorFlow / Keras
- Pandas / NumPy
- Matplotlib

## 🚀 Getting Started

```bash
git clone https://github.com/lina2016/bitcoin-prediction.git
cd bitcoin-prediction
pip install -r requirements.txt

bitcoin-prediction/
├── data/                  # Raw and processed CSV files
├── notebooks/             # Jupyter notebooks for exploration
├── models/                # Saved models and weights
├── utils/                 # Helper functions
├── main.py                # Training and prediction script
└── README.md              # Project overview




## 📂 Notebooks

Open the notebooks in order:

1. `data_processing.ipynb`
2. `model_LSTM.ipynb`
3. `prediction.ipynb`

---

## 📊 Model Overview

- **Architecture**: LSTM with sequence batching and dropout
- **Input**: Normalized Bitcoin price sequences
- **Output**: Predicted future prices with timestamp alignment
- **Evaluation**: Mean Squared Error (MSE), visual comparison with actual prices

---

## 🧪 Sample Output

| Date       | Predicted Price |
|------------|-----------------|
| 2025-08-05 | $29,842.17      |
| 2025-08-06 | $30,104.55      |

**Visualizations include:**

- Price trends over time
- Future forecast plot

---

## 📌 Author

**Lina** — AI TensorFlow Developer | Time Series Enthusiast
📍 Gilroy, CA
🔗 [LinkedIn Profile](https://www.linkedin.com/in/lina-jamadar/)

---

## 🧠 Future Work

- Compare LSTM with CNN, RNN, and DNN architectures
- Add hyperparameter tuning and model selection
- Deploy model via Flask or Streamlit for interactive forecasting

---

## 📜 License

This project is open-source and available under the **MIT License**.
