from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_FILE = Path("data/spam.csv")
MODEL_DIR = Path("model")
MODEL_FILE = MODEL_DIR / "spam_classifier.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"
REPORT_FILE = MODEL_DIR / "classification_report.txt"
SUMMARY_FILE = MODEL_DIR / "training_summary.txt"
SUBMISSION_REPORT = Path("Spam_Classifier_Report.rtf")
RANDOM_STATE = 42


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the Kaggle SMS spam CSV in the data folder."
        )

    df = pd.read_csv(path, encoding="latin-1")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    rename_map = {}
    if "v1" in df.columns:
        rename_map["v1"] = "label"
    if "v2" in df.columns:
        rename_map["v2"] = "message"
    df = df.rename(columns=rename_map)

    required_columns = {"label", "message"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'label' and 'message' columns, or the Kaggle-style 'v1' and 'v2' columns."
        )

    df = df[["label", "message"]].copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).apply(clean_text)
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"].str.len() > 0]
    df = df.drop_duplicates().reset_index(drop=True)
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def build_models() -> dict[str, Pipeline]:
    return {
        "MultinomialNB": Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
                ("classifier", MultinomialNB()),
            ]
        ),
        "LogisticRegression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "LinearSVC": Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
                ("classifier", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "spam_f1": f1_score(y_test, predictions, pos_label=1),
        "report": classification_report(y_test, predictions, target_names=["ham", "spam"]),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def select_best_model(
    models: dict[str, Pipeline], x_train: pd.Series, x_test: pd.Series, y_train: pd.Series, y_test: pd.Series
) -> tuple[str, Pipeline, dict, dict]:
    results = {}
    best_name = ""
    best_model = None
    best_metrics = None
    best_score = -1.0

    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        results[name] = {
            "accuracy": round(metrics["accuracy"], 4),
            "spam_f1": round(metrics["spam_f1"], 4),
            "confusion_matrix": metrics["confusion_matrix"],
        }

        if metrics["spam_f1"] > best_score:
            best_name = name
            best_model = model
            best_metrics = metrics
            best_score = metrics["spam_f1"]

    return best_name, best_model, best_metrics, results


def save_outputs(
    model_name: str,
    model: Pipeline,
    metrics: dict,
    comparison: dict,
    dataset: pd.DataFrame,
    test_size: int,
) -> dict:
    MODEL_DIR.mkdir(exist_ok=True)

    payload = {
        "model_name": model_name,
        "model": model,
        "labels": {0: "ham", 1: "spam"},
        "text_cleaning": "lowercase + url removal + punctuation removal + whitespace normalization",
    }
    joblib.dump(payload, MODEL_FILE)

    metrics_payload = {
        "best_model": model_name,
        "accuracy": round(metrics["accuracy"], 4),
        "spam_f1": round(metrics["spam_f1"], 4),
        "confusion_matrix": metrics["confusion_matrix"],
        "dataset_rows": int(len(dataset)),
        "ham_count": int((dataset["target"] == 0).sum()),
        "spam_count": int((dataset["target"] == 1).sum()),
        "test_samples": int(test_size),
        "model_comparison": comparison,
    }

    METRICS_FILE.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    REPORT_FILE.write_text(metrics["report"], encoding="utf-8")

    summary_lines = [
        "Spam Classifier Training Summary",
        f"Best model: {model_name}",
        f"Accuracy: {metrics_payload['accuracy']}",
        f"Spam F1-score: {metrics_payload['spam_f1']}",
        f"Dataset rows after cleaning: {metrics_payload['dataset_rows']}",
        f"Ham messages: {metrics_payload['ham_count']}",
        f"Spam messages: {metrics_payload['spam_count']}",
        f"Test samples: {metrics_payload['test_samples']}",
        "",
        "Model comparison:",
    ]
    for name, values in comparison.items():
        summary_lines.append(
            f"- {name}: accuracy={values['accuracy']}, spam_f1={values['spam_f1']}, confusion_matrix={values['confusion_matrix']}"
        )
    SUMMARY_FILE.write_text("\n".join(summary_lines), encoding="utf-8")
    return metrics_payload


def escape_rtf(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def write_submission_report(metrics_payload: dict) -> None:
    report_lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0 Calibri;}}",
        r"\fs24",
        r"\b Spam Classifier Project Report\b0\par",
        r"\par",
        r"This document summarizes the machine learning project developed to classify messages as spam or ham using text preprocessing and supervised classification algorithms.\par",
        r"\par",
        r"\b 1. Objective\b0\par",
        r"The objective of this project is to build a machine learning model that can automatically identify whether a text message is legitimate (ham) or unsolicited (spam).\par",
        r"\par",
        r"\b 2. Dataset\b0\par",
        r"The project uses the SMS Spam Collection dataset, a commonly used public dataset that is also shared through Kaggle mirrors. The dataset used in this project is stored as data/spam.csv.\par",
        rf"After cleaning, the dataset contained {metrics_payload['dataset_rows']} messages, with {metrics_payload['ham_count']} ham messages and {metrics_payload['spam_count']} spam messages.\par",
        r"\par",
        r"\b 3. Preprocessing\b0\par",
        r"The following preprocessing steps were applied before training the models:\par",
        r"1.\tab Convert message text to lowercase.\par",
        r"2.\tab Remove URLs.\par",
        r"3.\tab Remove punctuation and special characters.\par",
        r"4.\tab Normalize extra whitespace.\par",
        r"5.\tab Convert text into numerical features using TF-IDF with unigram and bigram features.\par",
        r"\par",
        r"\b 4. Algorithms Evaluated\b0\par",
        r"Three classification algorithms were trained and compared:\par",
        r"1.\tab Multinomial Naive Bayes\par",
        r"2.\tab Logistic Regression\par",
        r"3.\tab Linear Support Vector Classifier (LinearSVC)\par",
        r"\par",
        r"\b 5. Model Selection\b0\par",
        rf"The best model selected for this project was {escape_rtf(metrics_payload['best_model'])}. Model selection was based on the F1-score for the spam class, because spam detection should balance precision and recall.\par",
        rf"The selected model achieved an accuracy of {metrics_payload['accuracy']} and a spam F1-score of {metrics_payload['spam_f1']} on the test set.\par",
        rf"Confusion matrix: {escape_rtf(str(metrics_payload['confusion_matrix']))}\par",
        r"\par",
        r"\b 6. Output Files\b0\par",
        r"The project produces the following outputs:\par",
        r"1.\tab model/spam_classifier.joblib - saved trained model pipeline\par",
        r"2.\tab model/metrics.json - evaluation metrics and model comparison\par",
        r"3.\tab model/classification_report.txt - precision, recall and F1-score details\par",
        r"4.\tab app.py - prediction script for classifying new messages\par",
        r"\par",
        r"\b 7. How to Run\b0\par",
        r"1.\tab Install dependencies using pip install -r requirements.txt\par",
        r"2.\tab Train the model using python train.py\par",
        r"3.\tab Predict a message using python app.py \"Free tickets waiting for you\"\par",
        r"\par",
        r"\b 8. Conclusion\b0\par",
        r"This project demonstrates a complete natural language processing workflow for spam detection, including data loading, cleaning, feature extraction, model training, evaluation, and prediction. The final model can be reused to classify new text messages efficiently.\par",
        r"}",
    ]
    SUBMISSION_REPORT.write_text("\n".join(report_lines), encoding="ascii")


def main() -> None:
    df = load_dataset(DATA_FILE)
    x_train, x_test, y_train, y_test = train_test_split(
        df["message"],
        df["target"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    models = build_models()
    best_name, best_model, best_metrics, comparison = select_best_model(models, x_train, x_test, y_train, y_test)
    metrics_payload = save_outputs(best_name, best_model, best_metrics, comparison, df, len(x_test))
    write_submission_report(metrics_payload)

    print("Training completed successfully.")
    print(f"Best model: {best_name}")
    print(f"Accuracy: {metrics_payload['accuracy']}")
    print(f"Spam F1-score: {metrics_payload['spam_f1']}")
    print(f"Saved model to: {MODEL_FILE}")
    print(f"Saved metrics to: {METRICS_FILE}")
    print(f"Saved report to: {SUBMISSION_REPORT}")


if __name__ == "__main__":
    main()
