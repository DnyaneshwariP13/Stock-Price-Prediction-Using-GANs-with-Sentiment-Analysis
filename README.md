# Stock-Price-Prediction-Using-GANs-with-Sentiment-Analysis

# Stock Price Prediction Using GANs with Sentiment Analysis

This project leverages Generative Adversarial Networks (GANs) combined with sentiment analysis of tweets to predict stock prices. It integrates historical stock data with sentiment scores from Twitter to build a robust predictive model for stock price movements.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Model Overview](#model-overview)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to predict stock prices using a combination of historical stock data and sentiment analysis of tweets related to the stock. It employs a Generative Adversarial Network (GAN) for generating realistic stock price sequences and a Bidirectional Long Short-Term Memory (BiLSTM) network for sentiment analysis.

## Installation

To run this project, you'll need to install the required libraries. You can do this using pip:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn tqdm nltk statsmodels
```

Additionally, you'll need to download the VADER sentiment analysis tool from NLTK:

```python
import nltk
nltk.download('vader_lexicon')
```

## Usage

1. **Prepare Data**: Ensure you have the stock and tweet data in CSV format.
2. **Run the Script**: Execute the provided Python script to train the model and generate predictions.
3. **Visualize Results**: The script will produce plots showing the predicted vs. actual stock prices.


## Key Features

1. **Data Import and Preparation**
    - Reads stock and tweet data from CSV files.
    - Filters tweets for the specified stock.
    - Computes sentiment scores using VADER sentiment analysis.
    - Merges sentiment scores with stock data.

2. **Technical Indicators**
    - Computes various technical indicators (e.g., Moving Averages, MACD, Bollinger Bands).

3. **Data Normalization and Batching**
    - Normalizes data using MinMaxScaler.
    - Batches data for training the GAN.

4. **GAN Implementation**
    - Implements a generator model using LSTM layers.
    - Implements a discriminator model using Conv1D layers.
    - Defines loss functions and training steps for the GAN.

5. **Training and Evaluation**
    - Trains the GAN on the prepared dataset.
    - Evaluates the model on test data and plots the results.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - `numpy` for numerical computations
  - `pandas` for data manipulation
  - `matplotlib` for plotting
  - `tensorflow` and `keras` for deep learning
  - `scikit-learn` for preprocessing
  - `nltk` for sentiment analysis
  - `tqdm` for progress tracking

## Model Overview

### Generator Model

The generator is built using multiple LSTM layers to capture temporal dependencies in the stock price data. It aims to generate realistic stock price sequences conditioned on the input data.

### Discriminator Model

The discriminator is a convolutional neural network (CNN) designed to distinguish between real and generated stock price sequences. It helps the generator improve by providing feedback on the realism of generated sequences.

### Sentiment Analysis

The sentiment analysis component uses VADER from the NLTK library to assign sentiment scores (positive, neutral, negative, and compound) to tweets. These scores are integrated with the stock data to provide additional context for price movements.

## Results

The project evaluates the model's performance using metrics like Root Mean Squared Error (RMSE) and plots the predicted vs. actual stock prices. The visualizations demonstrate the model's ability to capture trends and fluctuations in stock prices.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to suggest improvements or report bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README provides an overview of the project, its structure, and usage. For detailed implementation, please refer to the code in the `scripts` directory.
