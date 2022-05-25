import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from tabulate import tabulate
from data_gen import BertSemanticDataGenerator

# loading SNLI trained model
model = tf.keras.models.load_model("./NLI_model")
print("SNLI trained model loaded...")

# labels
labels = ["contradiction", "entailment", "neutral"]

# loading test data
df = pd.read_csv(
    "SNLI_Corpus/snli_1.0_test.csv",
    low_memory=False,
    usecols=[
        "sentence1",
        "sentence2",
        "similarity"
    ]
)
df.dropna(axis=0, inplace=True)
df = df[df.similarity != "-"].sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"SNLI test data loaded...\n -shape: {df.shape}")


def check_similarity(sentence1, sentence2):
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )

    proba = model.predict(test_data[0])[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]*100: .2f}%"
    pred = labels[idx]
    return pred, proba


# inference
result = [
    check_similarity(premise, hypothesis)
    for premise, hypothesis in zip(df.sentence1.tolist(), df.sentence2.tolist())
]
df["prediction"] = list(map(lambda x: x[0], result))
df["score"] = list(map(lambda x: x[1], result))
print("inference done...")

# classification report
report = metrics.classification_report(y_true=df.similarity.tolist(), y_pred=df.prediction.tolist(), output_dict=True)
print(f"Accuracy: {round(report['accuracy'], 2)}", file=open("./classification_report.txt", "a"))
print(f"Macro Avg Precision: {round(report['macro avg'].get('precision'), 2)}", file=open("./classification_report.txt", "a"))
print(f"Weighted Avg Precision: {round(report['weighted avg'].get('precision'), 2)}", file=open("./classification_report.txt", "a"))

report = pd.DataFrame([
    {
        "label": key,
        "precision": value["precision"],
        "recall": value["recall"],
        "support": value["support"]
    }
    for key, value in report.items()
    if key not in ["accuracy", "macro avg", "weighted avg"]
])
print(
    tabulate(
        report,
        headers="keys",
        tablefmt="psql"
    ),
    file=open("./classification_report.txt", "a")
)
