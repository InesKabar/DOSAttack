# DoS Attack Detection

A Python machine learning project that uses a Decision Tree Classifier to detect Denial of Service (DoS) attacks.

## Project Overview

The project combines normal network traffic data with DoS attack data and trains a Decision Tree model to classify network activity as either Normal or DoS.

The dataset is divided into training and test sets. The model is then evaluated using a confusion matrix, accuracy, recall, and F1 score.

## Machine Learning Process

1. Load the Normal and DoS datasets.
2. Add labels to identify Normal and DoS traffic.
3. Combine the datasets.
4. Split the data into training and test sets.
5. Train a Decision Tree Classifier.
6. Predict the test set results.
7. Evaluate the model using classification metrics.
8. Visualize the decision tree.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Algorithm

- Decision Tree Classifier
- Entropy criterion
- Maximum depth: 3
- Minimum samples per leaf: 5
