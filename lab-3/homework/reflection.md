# Reflection – Lab 3: Audio AI Pipeline

**What I built.**  
I implemented an end-to-end audio pipeline: Speech-to-Text (STT) → multi-signal confidence scoring → PII redaction → summarization → Text-to-Speech (TTS), with audit logging for each stage. The pipeline accepts an MP3 input, but I also generated a clean 16 kHz mono WAV (“clean.wav”) to improve STT quality.

**What went well.**  
After fixing Google Cloud authentication and linking project billing, STT began returning results. Joining **all** STT result chunks (not just `results[0]`) eliminated truncation. Switching to LINEAR16 config for WAV and letting Google infer MP3 encoding resolved decode issues. TTS reliably produced a summary MP3 once a non-empty transcript was available.

**Confidence & uncertainty.**  
I used three signals: (1) the API’s confidence (or average word-level confidence as a fallback), (2) SNR computed from the waveform, and (3) a normalized combination score. Initially the combined confidence was 0.00 because STT returned an empty transcript; after the fixes, confidence rose (e.g., ~0.65). This shows why **not** to trust model confidence alone: audio quality (SNR), formatting/encoding, and segmentation heavily influence outcomes, and scores can be missing or misleading. A composite, multi-signal approach is more robust.

**PII handling.**  
I redacted numbers (credit card formats), phone, email via regex, and PERSON/DATE via spaCy NER. The fake card number was detected and replaced with `[REDACTED_CREDIT_CARD]`. Redaction happens before saving the transcript, and audit.log records what was removed (counts only, not raw values) to preserve privacy.

**Challenges and fixes.**  
Biggest blockers: credentials, billing, and audio encoding. I resolved them by running `gcloud auth application-default login`, enabling `speech.googleapis.com` and `texttospeech.googleapis.com`, linking billing, converting MP3→WAV with ffmpeg, and updating the STT config. I also replaced deprecated timestamps with timezone-aware UTC.

**What I’d improve next.**  
Add streaming STT for partial results, retry/backoff + secondary STT provider, diarization (who spoke), better summarization (TextRank or an LLM), and UX that color-codes low-confidence words for human review. I’d also unit-test PII patterns and add metrics dashboards for accuracy/cost monitoring.

**Takeaway.**  
End-to-end reliability requires both ML quality and solid engineering (formats, retries, logging, and governance).
