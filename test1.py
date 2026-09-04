import os
import json
import numpy as np
import scipy.signal
import Levenshtein
import soundfile as sf

import pocketsphinx
from pocketsphinx import Decoder, Config
from vosk import Model, KaldiRecognizer

# ------------------------------------------------------------------
# 1. PATH DEFINITIONS
# ------------------------------------------------------------------
VOSK_MODEL_DIR = r"C:\Users\HP\Documents\popi\vosk-model-small-en-us-0.15"

# Automatically resolve PocketSphinx Phone Language Model path
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
# 2. INITIALIZE ENGINES (< 90 MB TOTAL RAM)
# ------------------------------------------------------------------
print("⏳ Initializing Tier 1 Engine: Vosk Small (Kaldi C++)...")
if not os.path.exists(VOSK_MODEL_DIR):
    raise FileNotFoundError(f"Missing Vosk model folder: {VOSK_MODEL_DIR}")
vosk_model = Model(VOSK_MODEL_DIR)

print("⏳ Initializing Tier 2 Engine: PocketSphinx (Phoneme Decoder)...")
ps_config = Config()
if PHONE_LM_PATH:
    ps_config.set_string('-allphone', PHONE_LM_PATH)
    print(f"✅ Loaded PocketSphinx Phone LM from: {PHONE_LM_PATH}")
else:
    print("⚠️ Phone LM file not found! Falling back to unconstrained phone loop.")

ps_config.set_float('-lw', 2.5)           # Language weight for phoneme transition
ps_config.set_float('-samprate', 16000.0)
ps_config.set_string('-lm', None)         # Disable word-level LM

ps_decoder = Decoder(ps_config)

# ------------------------------------------------------------------
# 3. WORD CONFIGURATION MAP (SLP Clinical Targets)
# ------------------------------------------------------------------
WORD_CONFIG = {
    "sun": {
        "grammar": '["sun", "thun", "tun", "fun", "[unk]"]',
        "target_phonemes": ["S", "AH", "N"],
        "known_subs": {
            "TH": "Put your tongue behind your front teeth, not between them!",
            "T": "Don't press your tongue against the roof of your mouth!",
            "F": "Use your tongue against your teeth, not your top teeth on lower lip!"
        }
    },
    "rabbit": {
        "grammar": '["rabbit", "wabbit", "labbit", "[unk]"]',
        "target_phonemes": ["R", "AE", "B", "IH", "T"],
        "known_subs": {
            "W": "Curl your tongue back! Don't round your lips like a W sound.",
            "L": "Touch the roof of your mouth lightly with your tongue tip!",
            "B": "Make sure your tongue moves back, don't just use your lips!"
        }
    }
}

# ------------------------------------------------------------------
# 4. AUDIO PREPROCESSING UTILITY
# ------------------------------------------------------------------
def load_and_preprocess_audio(wav_path, target_sr=16000):
    data, original_sr = sf.read(wav_path, dtype='float32')

    # Convert stereo to mono if necessary
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample to 16,000 Hz if necessary
    if original_sr != target_sr:
        num_target_samples = int(len(data) * target_sr / original_sr)
        data = scipy.signal.resample(data, num_target_samples)

    # Convert float32 array to int16 PCM byte stream
    int16_bytes = (data * 32767).astype(np.int16).tobytes()
    return int16_bytes, target_sr

# ------------------------------------------------------------------
# 5. DUAL-TIER EVALUATION PIPELINE
# ------------------------------------------------------------------
def analyze_speech(wav_path, target_word):
    if not os.path.exists(wav_path):
        return {"file": os.path.basename(wav_path), "error": "File not found"}

    if target_word not in WORD_CONFIG:
        return {"error": f"Target word '{target_word}' not configured."}

    cfg = WORD_CONFIG[target_word]
    raw_audio_bytes, sample_rate = load_and_preprocess_audio(wav_path, target_sr=16000)

    # ==========================================
    # TIER 1: VOSK FAST-PATH (WITH CONFIDENCE GATE)
    # ==========================================
    rec = KaldiRecognizer(vosk_model, sample_rate, cfg["grammar"])
    rec.SetWords(True)  # Enable word-level confidence scoring
    rec.AcceptWaveform(raw_audio_bytes)
    res = json.loads(rec.FinalResult())
    
    detected_word = res.get("text", "").strip()
    words_info = res.get("result", [])
    
    confidence = words_info[0].get("conf", 0.0) if words_info else 0.0

    # FIX: Require BOTH target word match AND high confidence (>= 0.85)
    if detected_word == target_word and confidence >= 0.85:
        return {
            "file": os.path.basename(wav_path),
            "target": target_word,
            "status": "CORRECT",
            "tier": 1,
            "confidence": round(confidence, 2),
            "feedback": "Praise Path: Excellent articulation!"
        }

    # ==========================================
    # TIER 2: POCKETSPHINX PHONEME ALIGNMENT
    # ==========================================
    ps_decoder.start_utt()
    ps_decoder.process_raw(raw_audio_bytes, False, True)
    ps_decoder.end_utt()

    detected_phones = []
    if ps_decoder.hyp() is not None:
        for seg in ps_decoder.seg():
            phone = seg.word.upper()
            if phone not in ['SIL', '+NOISE+', '+SPEECH+', 'SILB', 'SILE', '<SIL>', '[SIL]', '+NSN+']:
                detected_phones.append(phone)

    target_str = "".join(cfg["target_phonemes"])
    detected_str = "".join(detected_phones)
    
    # Precise positional edit distance
    ops = Levenshtein.editops(target_str, detected_str)

    # FIX: Map edit operations directly to exact target position mismatch
    matched_sub = None
    hint = f"Let's practice saying '{target_word}' again!"

    if ops:
        # Find first edit operation (replace or insert)
        for op, t_idx, d_idx in ops:
            if op in ['replace', 'insert', 'delete']:
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
        "tier1_word_guess": detected_word,
        "tier1_confidence": round(confidence, 2),
        "detected_phonemes": detected_phones,
        "target_phonemes": cfg["target_phonemes"],
        "detected_substitution": matched_sub,
        "edit_distance": len(ops),
        "feedback_hint": hint
    }

# ------------------------------------------------------------------
# 6. BENCHMARK SUITE
# ------------------------------------------------------------------
if __name__ == "__main__":
    test_suite = [
        (r".\sun.wav", "sun"),
        (r".\thun.wav", "sun"),
        (r".\rabbit.wav", "rabbit"),
        (r".\wabbit.wav", "rabbit")
    ]

    print("\n🚀 RUNNING POPI PI-ZERO OPTIMIZED BENCHMARK:\n")
    for wav_path, target in test_suite:
        res = analyze_speech(wav_path, target)
        print(json.dumps(res, indent=2))
        print("-" * 50)