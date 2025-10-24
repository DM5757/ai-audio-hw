import os, sys, time, json, math, re
from datetime import datetime, timezone

import numpy as np
import librosa
import spacy
from google.cloud import speech, texttospeech


# ------------- Utility Functions -------------

def calculate_snr(audio_path):
    """Compute a simple SNR proxy in dB using 16k mono audio."""
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    if y.size == 0:
        return 0.0
    # Signal power and a crude noise estimate (residual after harmonic extraction)
    signal_power = float(np.mean(y ** 2))
    residual = y - librosa.effects.harmonic(y)
    noise_power = float(np.var(residual))
    if not np.isfinite(noise_power) or noise_power <= 0:
        noise_power = 1e-12
    snr_db = 10.0 * math.log10(signal_power / noise_power)
    # Clamp to a sane range to avoid blowing up normalization
    return float(np.clip(snr_db, -20.0, 60.0))


def summarize_text(text, max_sentences=2):
    """Very simple extractive summary: first N sentences."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    summary = ' '.join(sentences[:max_sentences]) if sentences else text.strip()
    return summary if summary.endswith(('.', '!', '?')) else summary + '.'


def clamp01(x):
    try:
        xf = float(x)
        if not np.isfinite(xf): return 0.0
        return 0.0 if xf < 0 else 1.0 if xf > 1 else xf
    except Exception:
        return 0.0


# ------------- PII Redaction -------------

PII_PATTERNS = {
    "CREDIT_CARD": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "PHONE":       r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
}

nlp = spacy.load("en_core_web_sm")

def redact_pii(text):
    redacted = text
    redactions = []

    # Regex pass
    for name, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, redacted):
            original = match.group(0)
            token = f"[REDACTED_{name}]"
            redacted = redacted.replace(original, token)
            redactions.append({"type": name, "value": original})

    # NER pass for names/dates
    doc = nlp(redacted)
    # Replace from right to left so offsets don’t shift
    ents = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
    for ent in ents:
        if ent.label_ in ["PERSON", "DATE"]:
            token = f"[REDACTED_{ent.label_}]"
            redactions.append({"type": ent.label_, "value": ent.text})
            redacted = redacted[:ent.start_char] + token + redacted[ent.end_char:]

    return redacted, redactions


# ------------- STT (joins all chunks) -------------

def transcribe_audio(audio_path):
    """Transcribe audio and return (full_transcript, api_conf_avg, all_words)."""
    client = speech.SpeechClient()

    with open(audio_path, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)
    ext = os.path.splitext(audio_path)[1].lower()

    if ext == ".wav":
        # Explicit LINEAR16 config for clean 16k WAVs
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True,
            enable_word_confidence=True,
            enable_word_time_offsets=True,
            use_enhanced=True,
            model="latest_long",
        )
    else:
        # MP3/other: let API infer encoding & rate
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
            language_code="en-US",
            enable_automatic_punctuation=True,
            enable_word_confidence=True,
            enable_word_time_offsets=True,
            use_enhanced=True,
            model="latest_long",
        )

    resp = client.recognize(config=config, audio=audio)
    if not resp.results:
        return "", 0.0, []

    all_texts, all_words, confs = [], [], []
    for res in resp.results:
        alt = res.alternatives[0]
        if alt.transcript:
            all_texts.append(alt.transcript)
        if getattr(alt, "words", None):
            all_words.extend(alt.words)
        if alt.confidence is not None:
            confs.append(float(alt.confidence))

    transcript = " ".join(all_texts).strip()
    api_conf = float(np.mean(confs)) if confs else 0.0
    return transcript, api_conf, all_words


# ------------- TTS -------------

def text_to_speech(text, output_file="output_summary.mp3"):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-A"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    resp = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    with open(output_file, "wb") as out:
        out.write(resp.audio_content)
    return output_file


# ------------- Main Pipeline -------------

def main(audio_path):
    start = time.time()
    with open("audit.log", "a", encoding="utf-8") as log_file:

        def log(event, **data):
            log_file.write(json.dumps({
                "time": datetime.now(timezone.utc).isoformat(),
                "event": event, **data
            }) + "\n")
            log_file.flush()

        log("start", file=audio_path)

        # Step 1: STT
        transcript, api_conf, words = transcribe_audio(audio_path)
        log("stt_done", chars=len(transcript), api_conf=api_conf)

        # Step 2: Confidence (robust)
        snr_db = calculate_snr(audio_path)
        # Map ~10..30 dB → 0..1 (clamped)
        snr_norm = clamp01((snr_db - 10.0) / 20.0)

        word_confs = [float(w.confidence) for w in words if getattr(w, "confidence", None) is not None]
        word_conf_avg = float(np.mean(word_confs)) if word_confs else 0.0

        primary_conf = api_conf if api_conf > 0 else word_conf_avg
        combined_conf = 0.5 * clamp01(primary_conf) + 0.3 * snr_norm + 0.2 * clamp01(word_conf_avg)
        level = "HIGH" if combined_conf > 0.85 else ("MEDIUM" if combined_conf > 0.70 else "LOW")

        log("confidence", snr_db=float(snr_db), snr_norm=float(snr_norm),
            api_conf=float(api_conf), word_conf_avg=float(word_conf_avg),
            combined=float(combined_conf), level=level)

        # Step 3: PII
        redacted, redactions = redact_pii(transcript)
        with open("output_transcript.txt", "w", encoding="utf-8") as f:
            f.write(redacted)
        log("pii_redaction", count=len(redactions))

        # Step 4: Summarize + TTS
        summary = summarize_text(redacted)
        tts_file = text_to_speech(summary)
        log("tts_done", output=tts_file, summary_chars=len(summary))

        log("done", elapsed=round(time.time() - start, 2))

    print("\n✅ Pipeline complete!")
    print(f"→ Transcript: output_transcript.txt")
    print(f"→ Summary: {tts_file}")
    print(f"→ Audit Log: audit.log")
    print(f"→ Confidence: {combined_conf:.2f} ({level})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_pipeline.py <audio_file>")
        sys.exit(1)
    main(sys.argv[1])
