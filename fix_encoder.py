"""
fix_encoder.py  — runs in under a second, no audio loading, no training.
Scans TESS filenames to get the same label set, fits a fresh OneHotEncoder,
and saves it as encoder.joblib compatible with the current numpy/sklearn.
"""
import os, numpy as np, joblib
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "final_sem_end_proj (2)", "final_sem_end_proj",
    "TESS Toronto emotional speech set data"
)
ENCODER_OUT = os.path.join(
    os.path.dirname(__file__),
    "final_sem_end_proj (2)", "final_sem_end_proj",
    "encoder.joblib"
)

def normalise(raw):
    raw = raw.lower().strip()
    m = {
        "ps": "ps", "pleasant_surprise": "ps", "pleasant_surprised": "ps",
        "surprised": "ps", "surprise": "ps",
        "anger": "angry", "angry": "angry",
        "disgust": "disgust", "fear": "fear",
        "happy": "happy", "happiness": "happy",
        "neutral": "neutral", "sad": "sad", "sadness": "sad",
    }
    return m.get(raw, raw)

labels = set()
for dirpath, _, files in os.walk(DATA_PATH):
    for fn in files:
        if fn.lower().endswith(".wav"):
            raw = fn.split("_")[-1].split(".")[0]
            labels.add(normalise(raw))

labels = sorted(labels)
print("Labels found:", labels)

enc = OneHotEncoder(sparse_output=False)
enc.fit(np.array(labels).reshape(-1, 1))
print("Encoder categories:", enc.categories_)

joblib.dump(enc, ENCODER_OUT)
print("Saved →", ENCODER_OUT)
