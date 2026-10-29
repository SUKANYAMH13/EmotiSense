"""
api.py — Unified Flask REST API for:
  • POST /predict/facial  — Facial Emotion Recognition (XCeption + dlib)
  • POST /predict/text    — Text Emotion Detection (BERT)
  • POST /predict/audio   — Speech Emotion Recognition (LSTM + MFCC)

Place this file in:  text_and_image/api.py
Run with:           python api.py
"""

import os
import sys
import io
import base64
import numpy as np
import subprocess
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
NEW_FACIAL_DIR = r"C:\Users\ADC\Downloads\MultimodalProject\MultimodalProject\video\Models"
TEXT_DIR   = os.path.join(BASE_DIR, "text_emotions_detection-main", "my-annoying-shrink-app")
AUDIO_DIR  = os.path.join(BASE_DIR, "final_sem_end_proj (2)", "final_sem_end_proj")

# Add text model's directory to path so myfuncs can be imported
sys.path.insert(0, TEXT_DIR)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the HTML frontend

# ─── Serve the frontend ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/style.css")
def serve_css():
    return send_from_directory(BASE_DIR, "style.css")

@app.route("/app.js")
def serve_js():
    return send_from_directory(BASE_DIR, "app.js")

# ═══════════════════════════════════════════════════════════════════════════════
# FACIAL EMOTION MODEL
# ═══════════════════════════════════════════════════════════════════════════════
import cv2
import dlib
from imutils import face_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from scipy.ndimage import zoom

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize dlib's face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
predictor_landmarks = dlib.shape_predictor(os.path.join(NEW_FACIAL_DIR, "face_landmarks.dat"))

# New XCeption model from the provided MultimodalProject path
facial_model = load_model(os.path.join(NEW_FACIAL_DIR, "video.h5"))
print("[✓] New XCeption Facial Emotion model loaded from external path.")
print("[✓] dlib face detector and landmarks predictor loaded.")


def predict_facial(image_bytes: bytes) -> dict:
    """Run dlib face detection + emotion classification using XCeption model."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use dlib to detect faces
    rects = face_detector(gray, 0)

    if len(rects) == 0:
        return {"emotion": None, "confidence": 0, "all_scores": {}, "message": "No face detected"}

    results = []
    for (i, rect) in enumerate(rects):
        # Identify face coordinates
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        
        # Ensure coordinates are within frame boundaries
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            continue

        roi_gray = gray[y:y+h, x:x+w]
        
        try:
            # XCeption preprocessing: zoom to 48x48
            # Replicating logic from live_face.py in the new repository
            roi_gray = zoom(roi_gray, (48 / roi_gray.shape[0], 48 / roi_gray.shape[1]))
            roi_gray = roi_gray.astype(np.float32)
            
            # Normalization: roi / float(roi.max())
            roi_max = float(roi_gray.max())
            if roi_max > 0:
                roi_gray /= roi_max
            
            # Reshape for model input (1, 48, 48, 1)
            roi = np.reshape(roi_gray.flatten(), (1, 48, 48, 1))
            
            preds = facial_model.predict(roi, verbose=0)[0]
            emotion_idx = int(np.argmax(preds))
            top_conf = float(preds[emotion_idx])

            all_scores = {EMOTION_LABELS[j]: round(float(preds[j]) * 100, 2) for j in range(len(EMOTION_LABELS))}
            results.append({
                "emotion":     EMOTION_LABELS[emotion_idx],
                "confidence":  round(top_conf * 100, 2),
                "all_scores":  all_scores,
                "box":         {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            })
        except Exception as e:
            print(f"Error processing face {i}: {e}")
            continue

    if not results:
        return {"emotion": None, "confidence": 0, "all_scores": {}, "message": "Face processing failed (could not process detected faces)"}

    # Return the face with highest confidence
    try:
        best = max(results, key=lambda r: r["confidence"])
        return best
    except ValueError:
        return {"emotion": None, "confidence": 0, "all_scores": {}, "message": "Calculation failed"}


@app.route("/predict/facial", methods=["POST"])
def facial_endpoint():
    """
    Accepts either:
      • multipart/form-data with key 'image' (file upload)
      • JSON { "image": "<base64 string>" }  (webcam frame)
    """
    try:
        if request.content_type and "multipart" in request.content_type:
            if "image" not in request.files:
                return jsonify({"error": "No image file provided"}), 400
            image_bytes = request.files["image"].read()
        else:
            data = request.get_json(force=True)
            b64 = data.get("image", "")
            if b64.startswith("data:image"):          # strip data-URL prefix
                b64 = b64.split(",", 1)[1]
            image_bytes = base64.b64decode(b64)

        result = predict_facial(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT EMOTION MODEL (BERT)
# ═══════════════════════════════════════════════════════════════════════════════
import myfuncs
from transformers import TFBertModel, BertTokenizerFast, BertConfig
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.initializers import TruncatedNormal

GE_TAXONOMY = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise"
]

EMOTION_META = {
    "admiration":     ["🤩", "\"You always admire what you really don't understand.\""],
    "amusement":      ["🥳", "\"The unfortunate who has to travel for amusement lacks capacity for amusement.\""],
    "anger":          ["😡", "\"Bitterness is like cancer. It eats upon the host. But anger is like fire. It burns it all clean.\""],
    "annoyance":      ["😤", "\"It could be worse.\""],
    "approval":       ["👍", "\"A man cannot be comfortable without his own approval.\""],
    "caring":         ["😘", "\"The road to hell is paved with good intentions.\""],
    "confusion":      ["🤨", "\"It's ok to be confused, but the most important thing is to know why...\""],
    "curiosity":      ["🤔", "\"The cure for boredom is curiosity. There is no cure for curiosity.\""],
    "desire":         ["😴", "\"From the deepest desire often come the deadliest hate.\""],
    "disappointment": ["😔", "\"You know the grass is always greener on the other side of the fence.\""],
    "disapproval":    ["🙄", "\"Don't judge a book by its cover.\""],
    "disgust":        ["🤮", "\"Some people are just beautifully wrapped boxes of trouble.\""],
    "embarrassment":  ["😳", "\"Shame is a soul eating emotion.\""],
    "excitement":     ["😆", "\"Excitement always leads to tears.\""],
    "fear":           ["😱", "\"Limits, like fears, are often just an illusion.\""],
    "gratitude":      ["🙏", "\"Revenge is profitable, gratitude is expensive.\""],
    "grief":          ["🪦", "\"All good things come to an end.\""],
    "joy":            ["🤗", "\"Ok then, everything is fine.\""],
    "love":           ["😍", "\"Love turns, with a little indulgence, to indifference or disgust.\""],
    "nervousness":    ["😬", "\"I wish my metabolism worked as fast as your anxiety.\""],
    "optimism":       ["💪", "\"Hope for the best, prepare for the worst.\""],
    "pride":          ["😎", "\"Among the blind, the one-eyed man is king.\""],
    "realization":    ["🥇", "\"Don't blow your own trumpet.\""],
    "relief":         ["😪", "\"Phew 😅\""],
    "remorse":        ["😞", "\"You can't unscramble a scrambled egg.\""],
    "sadness":        ["😭", "\"Don't worry, a couple of drinks and you will be just fine.\""],
    "surprise":       ["😮", "\"Sometimes I drink water just to surprise my liver.\""],
    "neutral":        ["😶", "\"So... 🤔\""]
}

# Build BERT model
bert_config  = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=False)
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', config=bert_config)

def _build_bert_model(nb_labels=27, max_length=48):
    transformer = TFBertModel.from_pretrained('bert-base-uncased', config=bert_config)
    bert_layer  = transformer.layers[0]
    input_ids      = Input(shape=(max_length,), name='input_ids',      dtype='int32')
    attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32')
    token_type_ids = Input(shape=(max_length,), name='token_type_ids', dtype='int32')
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
    bert_out  = bert_layer(inputs)[1]
    dropped   = Dropout(bert_config.hidden_dropout_prob, name='pooled_output')(bert_out, training=False)
    output    = Dense(nb_labels, activation='sigmoid',
                      kernel_initializer=TruncatedNormal(stddev=bert_config.initializer_range),
                      name='emotion')(dropped)
    return KerasModel(inputs=inputs, outputs=output, name='BERT_MultiLabel')

bert_model = _build_bert_model(27)
bert_model.load_weights(os.path.join(BASE_DIR, "text_emotions_detection-main", "bert-weights.hdf5"))
print("[✓] BERT Text Emotion model loaded.")


def predict_text(text: str, threshold: float = 0.92) -> dict:
    clean     = myfuncs.preprocess_corpus(text)
    tokenized = myfuncs.tokenize(bert_tokenizer, [clean])
    probas    = bert_model.predict(tokenized, verbose=0).ravel().tolist()

    labels       = [GE_TAXONOMY[i] for i, p in enumerate(probas) if p > threshold]
    proba_values = [p for p in probas if p > threshold]
    best_idx     = int(np.argmax(probas))
    best_label   = GE_TAXONOMY[best_idx]

    if not labels:
        labels       = ["neutral"]
        proba_values = [1.0]
        best_label   = "neutral"

    emotions_out = []
    for lbl, prob in zip(labels, proba_values):
        meta = EMOTION_META.get(lbl, ["❓", ""])
        emotions_out.append({
            "label":       lbl,
            "probability": round(prob * 100, 2),
            "emoji":       meta[0],
            "quote":       meta[1]
        })
    emotions_out.sort(key=lambda x: x["probability"], reverse=True)

    best_meta = EMOTION_META.get(best_label, ["❓", ""])
    return {
        "best_label":  best_label,
        "best_emoji":  best_meta[0],
        "best_quote":  best_meta[1],
        "emotions":    emotions_out
    }


@app.route("/predict/text", methods=["POST"])
def text_endpoint():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        result = predict_text(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# SPEECH / AUDIO EMOTION MODEL  (LSTM + MFCC)
# ═══════════════════════════════════════════════════════════════════════════════
import librosa
import joblib

AUDIO_EMOTION_META = {
    "angry":    ["😡", "Anger is like fire — it burns it all clean."],
    "disgust":  ["🤮", "Some people are just beautifully wrapped boxes of trouble."],
    "fear":     ["😱", "Limits, like fears, are often just an illusion."],
    "happy":    ["😄", "Happiness is not something ready-made. It comes from your own actions."],
    "neutral":  ["😶", "Stillness is where creativity and solutions are found."],
    "ps":       ["😮", "Surprise is the greatest gift which life can grant us."],
    "sad":      ["😭", "Even the darkest night will end and the sun will rise."],
}

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

try:
    # Rebuild exact architecture from train_model.py, then load weights only.
    # This bypasses the Keras-config version mismatch in the .h5 file.
    _enc_tmp = joblib.load(os.path.join(AUDIO_DIR, "encoder.joblib"))
    _num_classes = len(_enc_tmp.categories_[0])

    _m = Sequential([
        LSTM(256, return_sequences=False, input_shape=(40, 1)),
        Dropout(0.2),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(_num_classes, activation="softmax"),
    ])
    _m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    _m.load_weights(os.path.join(AUDIO_DIR, "emotion_model.h5"))

    audio_model   = _m
    audio_encoder = _enc_tmp
    print(f"[✓] Speech Emotion model loaded. ({_num_classes} classes: {list(_enc_tmp.categories_[0])})")
except Exception as e:
    audio_model   = None
    audio_encoder = None
    print(f"[✗] Speech Emotion model failed to load: {e}")



def predict_audio(audio_bytes: bytes) -> dict:
    """Extract MFCC features and predict emotion from raw audio bytes."""
    if audio_model is None or audio_encoder is None:
        return {"error": "Audio model not loaded"}
    try:
        # Transcode audio bytes to standard WAV using ffmpeg
        # This handles WebM/Opus from browsers or other compressed formats
        command = [
            'ffmpeg',
            '-y',                # Overwrite output files
            '-i', 'pipe:0',      # Input from stdin
            '-f', 'wav',         # Force output format to WAV
            '-acodec', 'pcm_s16le', # PCM 16-bit
            '-ar', '22050',      # Match common librosa default or model expectations
            '-ac', '1',          # Mono
            'pipe:1'             # Output to stdout
        ]
        
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        wav_bytes, stderr = process.communicate(input=audio_bytes)
        
        if process.returncode != 0:
            err_msg = stderr.decode('utf-8', errors='ignore')
            print(f"[✗] ffmpeg conversion failed: {err_msg}")
            # Fallback: maybe it's already a format librosa likes?
            wav_bytes = audio_bytes
        
        import io as _io
        y, sr = librosa.load(_io.BytesIO(wav_bytes), duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

        features = np.expand_dims(mfcc, axis=0)   # (1, 40)
        features = np.expand_dims(features, axis=-1) # (1, 40, 1)

        prediction = audio_model.predict(features, verbose=0)
        label      = audio_encoder.inverse_transform(prediction)[0][0].lower()
        confidence = float(np.max(prediction))

        labels       = audio_encoder.categories_[0]
        all_scores   = {lbl.lower(): round(float(prediction[0][i]) * 100, 2)
                        for i, lbl in enumerate(labels)}

        meta = AUDIO_EMOTION_META.get(label, ["🎙️", ""])
        return {
            "emotion":    label,
            "emoji":      meta[0],
            "quote":      meta[1],
            "confidence": round(confidence * 100, 2),
            "all_scores": all_scores
        }
    except Exception as e:
        return {"error": str(e)}


@app.route("/predict/audio", methods=["POST"])
def audio_endpoint():
    """
    Accepts multipart/form-data with key 'audio' (WAV / MP3 file)
    or JSON { "audio": "<base64 string>" }
    """
    try:
        if request.content_type and "multipart" in request.content_type:
            if "audio" not in request.files:
                return jsonify({"error": "No audio file provided"}), 400
            audio_bytes = request.files["audio"].read()
        else:
            data = request.get_json(force=True)
            b64  = data.get("audio", "")
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            audio_bytes = base64.b64decode(b64)

        result = predict_audio(audio_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("\n🚀 EmotiSense API running at http://127.0.0.1:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
