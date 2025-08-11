# Human Activity Recognition



## Project Overview

This project focuses on **Human Activity Recognition (HAR)**, a machine learning task that involves classifying human activities using data from a smartphone's accelerometer. We use a dataset containing accelerometer readings and pre-extracted statistical features to build and evaluate several machine learning and deep learning models. The goal is to accurately predict six different human activities: **walking, jogging, sitting, standing, upstairs, and downstairs.**

The project was conducted as part of the Msc Data Science program at Liverpool John Moores University (2022-2023) by Dhanushka F. Baduge and Shivam Goel.

---

Repository Structure

**notebooks:** This folder contains all the Jupyter Notebook files (.ipynb) where the data exploration, preprocessing, model training, and evaluation were performed. Each notebook corresponds to a specific part of the analysis.

**data:** This directory stores all the datasets used in the project, including the training and test files for both the raw signals and the metadata.

---

## Data Description

The dataset consists of two primary types of files: `signals.csv` and `metadata.csv`. These files contain accelerometer data and extracted features, respectively.

### `signals.csv`
This file contains raw time-series data from the accelerometer.

| Column | Description |
| :--- | :--- |
| `user_snippet` | A unique identifier for each user and data snippet (e.g., `user_id_snippet_id`). |
| `timestamp` | The timestamp of the reading in milliseconds. |
| `x-axis` | Acceleration in the x-direction. Values are between -20 and 20, where 10 equals 1g (9.81 m/s²). |
| `y-axis` | Acceleration in the y-direction. |
| `z-axis` | Acceleration in the z-direction. |

### `metadata.csv`
This file contains statistical features extracted from the raw signals data and the target variable.

| Column | Description |
| :--- | :--- |
| `user_snippet` | Matches the `user_snippet` in `signals.csv` and serves as a key to merge data. |
| `activity` | The target column with one of six labels: `walking`, `jogging`, `sitting`, `standing`, `upstairs`, or `downstairs`. |
| `x-axis__FEATURE` | 10 statistical features (e.g., `mean`, `median`, `standard_deviation`, `maximum`, etc.) extracted from the x-axis acceleration. |
| `y-axis__FEATURE` | The same 10 features extracted from the y-axis acceleration. |
| `z-axis__FEATURE` | The same 10 features extracted from the z-axis acceleration. |

---

## Methodology

Our approach followed a standard machine learning workflow, starting with data exploration and culminating in model evaluation.

### 1. Exploratory Data Analysis (EDA)
We began by analyzing the dataset's structure, checking for null values, and visualizing the distribution of activities. We discovered that **walking** was the most frequent activity, while **standing** was the least frequent. We also used plots to inspect the probability distributions of the accelerometer axes.



### 2. Data Pre-processing
The data was pre-processed using `StandardScaler` from scikit-learn to center the data around the origin with a standard deviation of 1. This step is crucial for many machine learning algorithms to perform optimally.

### 3. Dimensionality Reduction
Since the pairs plots showed significant class overlap, we applied three dimensionality reduction techniques to visualize the data in a 2D space and identify potential clusters:
* **Principal Component Analysis (PCA)**: Found some distinctions but still showed heavy class overlap.
* **t-SNE (T-distributed Stochastic Neighbor Embedding)**: With a `perplexity` of 85, this technique provided a much better separation of clusters, particularly for the `sitting` and `standing` classes.
* **UMAP (Unified Manifold Approximation & Projection)**: This technique did not produce well-defined clusters for our dataset.

### 4. Machine Learning Models
We trained and evaluated a variety of machine learning models on the pre-processed data. The models were assessed on a test set from a Kaggle competition.

| Model | Private Score | Public Score |
| :--- | :--- | :--- |
| **Gradient Boosting Classifier** | **0.83024** | **0.83018** |
| Random Forest | 0.80092 | 0.81358 |
| Logistic Regression | 0.80246 | 0.80301 |
| K-nearest neighbor (KNN) | 0.78472 | 0.80905 |
| Support Vector Classifier (SVC) | 0.78549 | 0.79698 |
| Multi-Layer Perceptron (MLP) | 0.7824 | 0.78339 |
| Naïve Bayes | 0.76697 | 0.77132 |

### 5. Deep Neural Networks (DNN)

We explored Deep Neural Networks (DNNs) to improve performance. Using Google Colab and the TensorFlow library, we experimented with 30 different DNN architectures, varying the number of layers, neurons, epochs, and dropout rates.

The best-performing model (`CNN26`) achieved the highest score and featured a specific architecture:
* **1st Hidden Layer**: 350 neurons, ReLU activation, 25% dropout.
* **2nd Hidden Layer**: 100 neurons, ReLU activation, 20% dropout.
* **3rd Hidden Layer**: 50 neurons, ReLU activation, 20% dropout.
* **Output Layer**: 6 neurons, Softmax activation.
* **Epochs**: 50

| Model | Private Score | Public Score |
| :--- | :--- | :--- |
| **CNN26 (DNN)** | 0.83873 | **0.84603** |
| CNN22 (DNN) | 0.84876 | 0.84603 |
| CNN29 (DNN) | 0.81404 | 0.82867 |

The final DNN model (`CNN30`) was developed after the competition closed but showed even better cross-validation results, indicating its potential for higher accuracy.

---

## Conclusion

The project successfully used statistical features from accelerometer data to predict human activities. While several traditional machine learning models showed promising results, the **Deep Neural Network (DNN)** architecture consistently achieved the highest accuracy scores.

Ultimately, this project demonstrated that deep learning models are highly effective for complex, multi-class classification problems like Human Activity Recognition, especially when provided with a comprehensive set of features.

---
