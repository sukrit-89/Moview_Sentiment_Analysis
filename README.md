# 🎬 IMDB Movie Sentiment Analysis with RNN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**A deep learning project that analyzes movie reviews and predicts sentiment using Recurrent Neural Networks (RNN)**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model Architecture](#-model-architecture) • [Examples](#-examples)

</div>

---

## 📖 Overview

This project demonstrates an end-to-end implementation of sentiment analysis using a Simple RNN (Recurrent Neural Network) trained on the IMDB movie reviews dataset. The model classifies movie reviews as either **positive** or **negative** with a confidence score.

The project includes:
- 📊 **Complete training pipeline** in Jupyter Notebook
- 🔮 **Prediction module** for testing on custom reviews
- 🌐 **Interactive web application** built with Streamlit
- 💾 **Pre-trained model** ready for inference

---

## ✨ Features

- **Deep Learning Model**: Simple RNN architecture with embedding layer for sentiment classification
- **Pre-trained on IMDB Dataset**: Trained on 25,000 movie reviews for binary sentiment classification
- **Interactive Web Interface**: User-friendly Streamlit app for real-time sentiment prediction
- **Comprehensive Pipeline**: From data preprocessing to model deployment
- **Custom Text Processing**: Handles vocabulary encoding and sequence padding
- **Confidence Scoring**: Provides probability scores for predictions

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MovieSentiment_analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install required dependencies**
   ```bash
   pip install tensorflow numpy streamlit
   ```

---

## 💻 Usage

### Option 1: Web Application (Streamlit)

Launch the interactive web interface:

```bash
streamlit run main.py
```

Then open your browser and navigate to the displayed local URL (typically `http://localhost:8501`).

**How to use:**
1. Enter your movie review in the text area
2. Click the "Classify" button
3. View the sentiment prediction and confidence score

### Option 2: Jupyter Notebooks

#### Training the Model

Open and run `RnnProject.ipynb` to:
- Load and explore the IMDB dataset
- Build the RNN architecture
- Train the model on 25,000 movie reviews
- Evaluate model performance
- Save the trained model

#### Making Predictions

Open `prediction.ipynb` to:
- Load the pre-trained model
- Test custom movie reviews
- Analyze prediction scores
- Experiment with different text inputs

### Option 3: Python Script

Use the prediction functions directly in your Python code:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load the model
model = load_model('simple_rnn_imdb.keras')

# Your custom review
review = "This movie was absolutely amazing! Great plot and excellent acting."

# Preprocess and predict
# (Use the preprocess_text function from main.py)
```

---

## 🧠 Model Architecture

The sentiment analysis model consists of the following layers:

```
Model: "sequential"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │ Param #       │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ embedding (Embedding)           │ (None, 500, 128)       │ 1,280,000     │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ simple_rnn (SimpleRNN)          │ (None, 128)            │ 32,896        │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 1)              │ 129           │
└─────────────────────────────────┴────────────────────────┴───────────────┘
Total params: 1,313,025 (5.01 MB)
Trainable params: 1,313,025 (5.01 MB)
```

### Key Components:

- **Embedding Layer**: Converts word indices to dense vectors (128 dimensions)
- **Simple RNN Layer**: Processes sequential information with 128 hidden units
- **Dense Layer**: Outputs sentiment probability (sigmoid activation)
- **Vocabulary Size**: 10,000 most common words
- **Sequence Length**: 500 words (padded/truncated)

---

## 📝 Examples

### Positive Review Example

**Input:**
```
"This is a great movie and the plot was thrilling"
```

**Output:**
```
Sentiment: Positive
Prediction Score: 0.87
```

### Negative Review Example

**Input:**
```
"This movie was average and the acting was not so good"
```

**Output:**
```
Sentiment: Negative
Prediction Score: 0.32
```

---

## 📁 Project Structure

```
MovieSentiment_analysis/
│
├── RnnProject.ipynb          # Complete training pipeline and model building
├── prediction.ipynb          # Prediction testing and experimentation
├── main.py                   # Streamlit web application
├── simple_rnn_imdb.keras     # Pre-trained model (15 MB)
├── README.md                 # Project documentation
└── .gitignore                # Git ignore rules
```

---

## 🔧 Technical Details

### Dataset
- **Source**: IMDB Movie Reviews Dataset (via Keras)
- **Training Samples**: 25,000 labeled reviews
- **Test Samples**: 25,000 labeled reviews
- **Classes**: Binary (Positive/Negative)

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Max Features**: 10,000 words
- **Max Length**: 500 words per review

### Text Preprocessing
1. Convert text to lowercase
2. Tokenize into words
3. Map words to integer indices using IMDB word index
4. Handle out-of-vocabulary words (OOV)
5. Pad/truncate sequences to fixed length (500)

---

## 🎯 Performance

The model achieves competitive performance on the IMDB test set, effectively distinguishing between positive and negative sentiment in movie reviews.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **TensorFlow/Keras** | Deep learning framework for building and training the RNN |
| **NumPy** | Numerical computations and array operations |
| **Streamlit** | Interactive web application framework |
| **Python** | Primary programming language |

---

## 📚 Learning Objectives

This project demonstrates:

✅ Building and training RNN models for NLP tasks  
✅ Working with sequential data and text preprocessing  
✅ Implementing word embeddings  
✅ Creating end-to-end machine learning pipelines  
✅ Deploying models with interactive web interfaces  
✅ Handling text encoding and vocabulary management  

---

## 🤝 Contributing

This is a practice project, but suggestions and improvements are welcome! Feel free to:

- Report bugs or issues
- Suggest new features
- Improve documentation
- Optimize model performance

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **IMDB Dataset**: Provided by Andrew Maas et al.
- **TensorFlow/Keras**: For the excellent deep learning framework
- **Streamlit**: For the intuitive app development platform

---

## 📞 Contact

For questions or feedback about this practice project, feel free to reach out or open an issue.

---

<div align="center">

**Made with ❤️ as a deep learning practice project**

⭐ Star this repo if you found it helpful!

</div>
