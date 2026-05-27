# Spam Classifier

This project builds a machine learning model to classify text messages as `spam` or `ham`. It uses text preprocessing, TF-IDF feature extraction, and multiple classification algorithms to select the best-performing model.

## Project Structure

```text
spam-classifier/
|
+-- data/
|   +-- spam.csv
|
+-- model/
|   +-- spam_classifier.joblib
|   +-- metrics.json
|   +-- classification_report.txt
|   +-- training_summary.txt
|
+-- app.py
+-- train.py
+-- requirements.txt
+-- README.md
```

## Dataset

- Dataset: SMS Spam Collection
- Source type: Kaggle mirror / public SMS spam dataset
- Expected file: `data/spam.csv`
- Main columns used: `v1` or `label`, and `v2` or `message`

## Features

- Loads and cleans the dataset
- Removes empty rows, duplicates, and unused columns
- Applies text preprocessing
- Converts text to TF-IDF features
- Trains and compares `MultinomialNB`, `LogisticRegression`, and `LinearSVC`
- Selects the best model using spam F1-score
- Saves the trained model and evaluation files
- Predicts new messages from the command line
- Generates a submission-ready Word-compatible report file

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py
```

This creates:

- `model/spam_classifier.joblib`
- `model/metrics.json`
- `model/classification_report.txt`
- `model/training_summary.txt`
- `Spam_Classifier_Report.rtf`

## Run In Browser

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Predict New Messages

You can paste a message into the browser UI and click `Classify Message`.

## Submission Notes

- `Spam_Classifier_Report.rtf` opens in Microsoft Word
- The project is ready to push to GitHub once you share the repository link
