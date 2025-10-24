from google.cloud import speech, texttospeech
import sys


def summarize_text(text, max_sentences=3):
    """
    Simple extractive summarization: pick first N sentences.
    For homework, use better methods (TF-IDF, TextRank, or GPT)
    """
    sentences = text.split('. ')
    summary = '. '.join(sentences[:max_sentences])

    if not summary.endswith('.'):
        summary += '.'

    return summary


def text_to_speech(text, output_file="output_summary.mp3", voice_name="en-US-Neural2-A"):
    """Generate audio from text using Google Cloud TTS"""

    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=voice_name
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0,
        pitch=0.0,
        volume_gain_db=0.0,
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    with open(output_file, 'wb') as out:
        out.write(response.audio_content)

    print(f"Audio summary saved: {output_file}")
    return output_file


def transcribe_summarize_tts(audio_path):
    """Complete pipeline: STT -> Summarize -> TTS"""

    print("Step 1: Transcribing audio...")

    speech_client = speech.SpeechClient()

    with open(audio_path, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = speech_client.recognize(config=config, audio=audio)
    transcript = response.results[0].alternatives[0].transcript

    print(f"Original Transcript ({len(transcript)} chars):")
    print(transcript)

    print("Step 2: Summarizing...")
    summary = summarize_text(transcript, max_sentences=2)

    print(f"Summary ({len(summary)} chars):")
    print(summary)

    print("Step 3: Generating audio summary...")
    output_file = text_to_speech(summary)

    print("=" * 60)
    print("PIPELINE COMPLETE!")
    print(f"   Original audio: {audio_path}")
    print(f"   Summary audio: {output_file}")
    print(f"   Compression: {len(transcript)} -> {len(summary)} chars")
    print("=" * 60)

    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 4_tts_summary.py <audio_file>")
        sys.exit(1)

    transcribe_summarize_tts(sys.argv[1])
