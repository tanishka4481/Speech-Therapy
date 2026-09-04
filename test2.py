import os
import json
import numpy as np
import scipy.signal
import Levenshtein
import soundfile as sf

import sherpa_onnx
import pocketsphinx
from pocketsphinx import Decoder, Config

# ------------------------------------------------------------------
# 1. PATH CONFIGURATIONS
# ------------------------------------------------------------------
# Download lightweight Citrinet-512 or Conformer-Small CTC model from sherpa-onnx releases:
# https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
SHERPA_MODEL_DIR = r"C:\Users\HP\Documents\popi\sherpa-onnx-nemo-ctc-en-citrinet-512"

# Automatically locate PocketSphinx Phone Language Model
PS_DIR = os.path.dirname(pocketsphinx.__file__)
candidate_paths = [
    os.path.join(PS_DIR, 'model', 'en-us-phone.lm.bin'),
    os.path.join(PS_DIR, 'en-us-phone.lm.bin'),
    os.path.join(PS_DIR, 'model', 'en-us', 'en-us-phone.lm.bin')
]

PHONE_LM_PATH = None
for path in candidate_paths:
    if os.path.exists(path):
        PHONE_LM_PATH = path
        break

# ------------------------------------------------------------------
# 2. INITIALIZE DUAL ENGINES (< 120 MB RAM TOTAL)
# ------------------------------------------------------------------
print("⏳ Initializing Tier 1 Engine: Sherpa-ONNX (Lightweight CTC INT8)...")

# Check model paths (Supports Citrinet, Conformer-Small, or Parakeet CTC)
model_onnx = os.path.join(SHERPA_MODEL_DIR, "model.int8.onnx")
if not os.path.exists(model_onnx):
    model_onnx = os.path.join(SHERPA_MODEL_DIR, "model.onnx")

tokens_txt = os.path.join(SHERPA_MODEL_DIR, "tokens.txt")

if not os.path.exists(model_onnx) or not os.path.exists(tokens_txt):
    raise FileNotFoundError(
        f"Missing Sherpa-ONNX model files in {SHERPA_MODEL_DIR}.\n"
        f"Ensure 'model.int8.onnx' and 'tokens.txt' are extracted there."
    )

sherpa_recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
    model=model_onnx,
    tokens=tokens_txt,
    num_threads=2,             # Uses 2 cores on Pi Zero 2W
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search"
)

print("⏳ Initializing Tier 2 Engine: PocketSphinx (Phoneme Alignment)...")
ps_config = Config()
if PHONE_LM_PATH:
    ps_config.set_string('-allphone', PHONE_LM_PATH)
    print(f"✅ Loaded PocketSphinx Phone LM from: {PHONE_LM_PATH}")
else:
    print("⚠️ Phone LM file not found! Falling back to unconstrained phone loop.")

ps_config.set_float('-lw', 2.0)
ps_config.set_float('-samprate', 16000.0)
ps_config.set_string('-lm', None)

ps_decoder = Decoder(ps_config)

# ------------------------------------------------------------------
# 3. CLINICAL WORD CONFIGURATION MAP
# ------------------------------------------------------------------
WORD_CONFIG = {
    "sun": {
        "valid_matches": ["SUN", "SON"],
        "target_phonemes": ["S", "AH", "N"],
        "known_subs": {
            "TH": "Put your tongue behind your front teeth, not between them!",
            "T": "Don't press your tongue against the roof of your mouth!",
            "F": "Use your tongue against your teeth, not your top teeth on lower lip!"
        }
    },
    "rabbit": {
        "valid_matches": ["RABBIT", "RABBITS"],
        "target_phonemes": ["R", "AE", "B", "IH", "T"],
        "known_subs": {
            "W": "Curl your tongue back! Don't round your lips like a W sound.",
            "L": "Touch the roof of your mouth lightly with your tongue tip!",
            "B": "Make sure your tongue moves back, don't just use your lips!"
        }
    }
}

# ------------------------------------------------------------------
# 4. AUDIO NORMALIZATION HELPER (16KHZ MONO PCM)
# ------------------------------------------------------------------
def load_and_preprocess_audio(wav_path, target_sr=16000):
    data, sr = sf.read(wav_path, dtype='float32')
    
    # Convert stereo to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        
    # Resample to 16,000 Hz if necessary
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        data = scipy.signal.resample_poly(data, target_sr // gcd, sr // gcd)

    float_data = data.astype(np.float32)
    int16_bytes = (data * 32767).astype(np.int16).tobytes()

    return float_data, int16_bytes

# ------------------------------------------------------------------
# 5. DUAL-TIER EVALUATION PIPELINE
# ------------------------------------------------------------------
def analyze_speech(wav_path, target_word):
    if not os.path.exists(wav_path):
        return {"file": os.path.basename(wav_path), "error": "File not found"}

    if target_word not in WORD_CONFIG:
        return {"error": f"Target word '{target_word}' not configured."}

    cfg = WORD_CONFIG[target_word]
    float_data, raw_bytes = load_and_preprocess_audio(wav_path, target_sr=16000)

    # ==========================================
    # TIER 1: SHERPA-ONNX CTC DECODING
    # ==========================================
    stream = sherpa_recognizer.create_stream()
    stream.accept_waveform(16000, float_data)
    sherpa_recognizer.decode_stream(stream)
    
    sherpa_transcript = stream.result.text.upper().strip()
    words_heard = sherpa_transcript.split()

    # Exact string verification against valid target options
    is_correct = any(valid_word in words_heard for valid_word in cfg["valid_matches"])

    if is_correct:
        return {
            "file": os.path.basename(wav_path),
            "target": target_word,
            "status": "CORRECT",
            "tier": 1,
            "sherpa_transcript": sherpa_transcript,
            "feedback": "Praise Path: Excellent articulation!"
        }

    # ==========================================
    # TIER 2: POCKETSPHINX PHONEME ALIGNMENT
    # ==========================================
    ps_decoder.start_utt()
    ps_decoder.process_raw(raw_bytes, False, True)
    ps_decoder.end_utt()

    detected_phones = []
    if ps_decoder.hyp() is not None:
        for seg in ps_decoder.seg():
            phone = seg.word.upper()
            if phone not in ['SIL', '+NOISE+', '+SPEECH+', 'SILB', 'SILE', '<SIL>', '[SIL]', '+NSN+']:
                detected_phones.append(phone)

    target_str = "".join(cfg["target_phonemes"])
    detected_str = "".join(detected_phones)
    
    # Calculate exact positional Levenshtein edit operations
    ops = Levenshtein.editops(target_str, detected_str)

    matched_sub = None
    hint = f"Let's practice articulating '{target_word}' again!"

    # Map mismatches to corrective SLP hints
    if ops:
        for op, t_idx, d_idx in ops:
            if op in ['replace', 'insert']:
                if d_idx < len(detected_phones):
                    substituted_phone = detected_phones[d_idx]
                    if substituted_phone in cfg["known_subs"]:
                        matched_sub = substituted_phone
                        hint = cfg["known_subs"][substituted_phone]
                        break

    return {
        "file": os.path.basename(wav_path),
        "target": target_word,
        "status": "MISARTICULATION_DETECTED",
        "tier": 2,
        "sherpa_transcript": sherpa_transcript,
        "detected_phonemes": detected_phones,
        "target_phonemes": cfg["target_phonemes"],
        "detected_substitution": matched_sub,
        "edit_distance": len(ops),
        "feedback_hint": hint
    }

# ------------------------------------------------------------------
# 6. TEST BENCHMARK
# ------------------------------------------------------------------
if __name__ == "__main__":
    test_suite = [
        (r".\sun.wav", "sun"),
        (r".\thun.wav", "sun"),
        (r".\rabbit.wav", "rabbit"),
        (r".\wabbit.wav", "rabbit")
    ]

    print("\n🚀 RUNNING POPI SHERPA-ONNX + POCKETSPHINX BENCHMARK:\n")
    for wav_path, target in test_suite:
        res = analyze_speech(wav_path, target)
        print(json.dumps(res, indent=2))
        print("-" * 50)