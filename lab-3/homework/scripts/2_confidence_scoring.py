# scripts/2_confidence_scoring.py

from google.cloud import speech
import librosa
import numpy as np
import sys

def calculate_snr(audio_path):
    """Calculate Signal-to-Noise Ratio"""
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Estimate signal power (mean of squared samples)
    signal_power = np.mean(y ** 2)
    
    # Estimate noise power (variance)
    noise_power = np.var(y)
    
    # Calculate SNR in dB
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')  # No noise detected
    
    return snr_db


def calculate_word_perplexity(words):
    """
    Calculate perplexity based on word-level confidence.
    Lower confidence = higher perplexity = more uncertain
    """
    confidences = [word.confidence for word in words]
    avg_confidence = np.mean(confidences)
    
    # Convert to perplexity-like score (inverse of confidence)
    perplexity = 1.0 / avg_confidence if avg_confidence > 0 else float('inf')
    
    return perplexity


def multi_factor_confidence(audio_path):
    """Analyze confidence using multiple factors"""
    
    # Factor 1: Google's API confidence
    client = speech.SpeechClient()
    
    with open(audio_path, 'rb') as audio_file:
        content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_word_confidence=True,
    )
    
    response = client.recognize(config=config, audio=audio)
    
    # Extract API confidence and words
    api_confidence = response.results[0].alternatives[0].confidence
    words = response.results[0].alternatives[0].words
    transcript = response.results[0].alternatives[0].transcript
    
    # Factor 2: Audio Quality (SNR)
    snr = calculate_snr(audio_path)
    
    # Factor 3: Language Perplexity
    perplexity = calculate_word_perplexity(words)
    
    # Combined Score (weighted average)
    # API confidence: 50%, SNR: 30%, Perplexity: 20%
    
    # Normalize SNR to 0-1 scale (assume 10dB = good, 30dB = excellent)
    snr_normalized = min(max((snr - 10) / 20, 0), 1)
    
    # Normalize perplexity (lower is better, typical range 1-2)
    perplexity_normalized = max(1 - (perplexity - 1), 0)
    
    combined_score = (
        0.5 * api_confidence +
        0.3 * snr_normalized +
        0.2 * perplexity_normalized
    )
    
    # Determine confidence level
    if combined_score > 0.85:
        confidence_level = "🟢 HIGH"
    elif combined_score > 0.70:
        confidence_level = "🟡 MEDIUM"
    else:
        confidence_level = "🔴 LOW"
    
    # Print results
    print(f"\n{'='*60}")
    print(f"MULTI-FACTOR CONFIDENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Transcript: {transcript}")
    print(f"\nFactor Scores:")
    print(f"  API Confidence:    {api_confidence:.3f} (weight: 50%)")
    print(f"  Audio Quality (SNR): {snr:.2f} dB → {snr_normalized:.3f} (weight: 30%)")
    print(f"  Perplexity:        {perplexity:.3f} → {perplexity_normalized:.3f} (weight: 20%)")
    print(f"\n🎯 Combined Score:    {combined_score:.3f}")
    print(f"📊 Confidence Level: {confidence_level}")
    print(f"{'='*60}")
    
    if confidence_level == "🔴 LOW":
        print("\n🚧  WARNING: Low confidence. Manual review recommended.")
    
    return combined_score, confidence_level


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 2_confidence_scoring.py <audio_file>")
        sys.exit(1)
    
    multi_factor_confidence(sys.argv[1])
