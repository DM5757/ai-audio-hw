"""Step 1: Basic Speech‑to‑Text transcription.

This script demonstrates how to send an audio file to the Google Cloud
Speech‑to‑Text API and print the resulting transcript along with
confidence scores and word‑level timing information.  It aims to be
robust to different audio formats by inferring the encoding from the
filename.

Usage:
    python 1_basic_stt.py path/to/audio_file

The script prints the transcript, overall confidence and per‑word details.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from google.cloud import speech


def guess_encoding(file_path: str) -> speech.RecognitionConfig.AudioEncoding:
    """Infer the audio encoding from the file extension.

    If the extension is unrecognised, returns ENCODING_UNSPECIFIED, which
    allows the API to attempt to determine the encoding automatically.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".mp3":
        return speech.RecognitionConfig.AudioEncoding.MP3
    if ext in {".wav", ".pcm", ".raw"}:
        return speech.RecognitionConfig.AudioEncoding.LINEAR16
    return speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED


def transcribe_audio(audio_path: str) -> Optional[speech.RecognizeResponse]:
    """Transcribe the given audio file and print results.

    Returns the API response on success or None on failure.
    """
    # Read the binary content of the audio file
    try:
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()
    except FileNotFoundError:
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        return None
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error reading file: {exc}", file=sys.stderr)
        return None

    # Initialize API client
    client = speech.SpeechClient()

    # Build recognition request
    audio = speech.RecognitionAudio(content=content)
    encoding = guess_encoding(audio_path)
    config = speech.RecognitionConfig(
        encoding=encoding,
        language_code="en-US",
        enable_automatic_punctuation=True,
        enable_word_confidence=True,
        enable_word_time_offsets=True,
        model="default",
    )

    # Perform synchronous recognition (suitable for audio <1 minute)
    print(f"Transcribing {audio_path} using encoding {encoding.name}...")
    try:
        response = client.recognize(config=config, audio=audio)
    except Exception as exc:  # pylint: disable:broad-except
        print(f"API error: {exc}", file=sys.stderr)
        return None

    # Print results
    for result_index, result in enumerate(response.results):
        alternative = result.alternatives[0]
        transcript = alternative.transcript
        confidence = alternative.confidence
        print(f"\nResult {result_index + 1}:")
        print(f"Transcript: {transcript}")
        print(f"Confidence: {confidence:.3f}")
        print("\nWord‑level details:")
        for word_info in alternative.words:
            word = word_info.word
            word_conf = word_info.confidence if word_info.confidence is not None else 0.0
            start_time = word_info.start_time.total_seconds() if word_info.start_time else 0.0
            end_time = word_info.end_time.total_seconds() if word_info.end_time else 0.0
            print(f"  {word} | conf={word_conf:.3f} | [{start_time:.2f}s – {end_time:.2f}s]")

    return response


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python 1_basic_stt.py <audio_file>", file=sys.stderr)
        sys.exit(1)
    audio_path = sys.argv[1]
    transcribe_audio(audio_path)


if __name__ == "__main__":
    main()
