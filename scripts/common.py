"""Shared helpers for the revised bug-report experiments."""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier

SEED = 42

# ---------------------------------------------------------------------------
# Template / boilerplate removal (label-free, fixed rules)
# ---------------------------------------------------------------------------
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
HEADING = re.compile(r"^\s*#{1,6}\s.*$", re.M)
CHECKBOX = re.compile(r"^\s*[-*]\s*\[[ xX]\].*$", re.M)
# Prow slash commands set labels directly (/kind bug, /sig node ...)
SLASH_CMD = re.compile(
    r"^\s*/(kind|sig|area|priority|triage|remove-[\w-]+|assign|cc|milestone|label|lifecycle|wg|good-first-issue|help)\b.*$",
    re.M | re.I)
TEMPLATE_LINES = [
    r"_No response_",
    r"\$ cat /etc/os-release",
    r"\$ uname -a",
    r"\$ kubectl version",
    r"C:\\> wmic os get Caption, Version, BuildNumber, OSArchitecture",
    r"</?details>",
    r"</?summary>",
    r"Let's make sure issue is not already fixed in latest builds first\.",
    r"From a fresh installation and clean workspace:",
    r"From `<workspace>/\.metadata/\.log`",
]
CODE_FENCE = re.compile(r"```[\w+-]*")  # fence markers and language tags (```console)
TEMPLATE_LINE_RE = re.compile(r"^\s*(?:" + "|".join(TEMPLATE_LINES) + r")\s*$", re.M | re.I)


def strip_template(text: str) -> str:
    t = HTML_COMMENT.sub(" ", text)
    t = SLASH_CMD.sub(" ", t)
    t = HEADING.sub(" ", t)
    t = CHECKBOX.sub(" ", t)
    t = TEMPLATE_LINE_RE.sub(" ", t)
    t = CODE_FENCE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def load(path):
    df = pd.read_csv(path)
    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["text"] = (df["title"].str.strip() + " " + df["body"].str.strip()).str.strip()
    df["body_clean"] = df["body"].map(strip_template)
    df["text_clean"] = (df["title"].str.strip() + " " + df["body_clean"]).str.strip()
    df["strata"] = df["repo"] + "_" + df["is_bug"].astype(str)
    return df.reset_index(drop=True)


TEXT_VARIANTS = {
    "combined": "text",
    "combined_no_template": "text_clean",
    "title_only": "title",
    "description_only": "body",
    "description_no_template": "body_clean",
}


def vectorizer(min_df=5):
    # Settings used in the original analysis notebook
    return TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                           min_df=min_df, max_df=0.95, max_features=25000,
                           sublinear_tf=True)


def models():
    return {
        "Majority": DummyClassifier(strategy="most_frequent"),
        "Stratified random": DummyClassifier(strategy="stratified", random_state=SEED),
        "MNB": MultinomialNB(alpha=1.0),
        "LR": LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED),
        "Linear SVM": LinearSVC(C=1.0, random_state=SEED),
    }


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
