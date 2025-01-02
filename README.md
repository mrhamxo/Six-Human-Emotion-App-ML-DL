# Six Human Emotion classification App - Deep Learning & Machine Learning Models

A comprehensive Streamlit-based application that predicts human emotions from text using both **Deep Learning** and **Machine Learning** models. This app supports multiple emotion categories: `Joy`, `Fear`, `Anger`, `Love`, `Sadness`, and `Surprise`.

## 🚀 Features
- **Deep Learning Model**:
  - Leverages a pre-trained model (`dl_model.h5`) for advanced emotion detection.
  - Supports input sequence processing with vocabulary and sequence length configurations.
- **Machine Learning Model**:
  - Uses a Random Forest Classifier (`rfc_model.pkl`) and TF-IDF vectorizer (`tfidf_vectorizer.pkl`).
  - Suitable for lightweight and fast predictions.
- **Dynamic Model Selection**:
  - Allows users to choose between the Deep Learning or Machine Learning models.
- **User-Friendly Interface**:
  - Built with Streamlit for an interactive and intuitive user experience.

## 💡 Usage

1. Launch the app using the above command.
2. Enter a sentence in the input box.
3. Select the desired model (Deep Learning or Machine Learning) from the dropdown.
4. Click on the "Predict Emotion" button to see the predicted emotion and probability.

## 📊 Emotion Categories

The app predicts the following six emotions:
- **Joy**: "I feel strong and good overall."
- **Fear**: "I’m all alone in this world."
- **Anger**: "This is outrageous!"
- **Love**: "He is really sweet and caring."
- **Sadness**: "I feel like I’ve lost everything."
- **Surprise**: "He was speechless when he found out."

## 🧠 Models Overview

### 1. **Deep Learning**
- **Model File**: `dl_model.h5`
- **Description**: Uses an advanced neural network for high-accuracy emotion predictions.
- **Requirements**: Vocabulary size and maximum sequence length are loaded from `vocab_info.pkl`.

### 2. **Machine Learning**
- **Model File**: `rfc_model.pkl`
- **Description**: A Random Forest Classifier trained on TF-IDF features for fast predictions.

## 📧 Contact

For questions or feedback, feel free to reach out:
- **Name**: Muhammad Hamza
- **Email**: [mr.hamxa942@gmail.com](mailto:mr.hamxa942@gmail.com)
- **GitHub**: [mrhamza](https://github.com/mrhamxo)
