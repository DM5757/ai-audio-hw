# scripts/3_pii_redaction.py

from google.cloud import speech
import re
import spacy
import sys

# Load spaCy model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

def redact_pii_regex(text):
    """Redact PII using regex patterns"""
    
    patterns = {
        'CREDIT_CARD': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        'SSN': r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        'PHONE': r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    }
    
    redacted = text
    redactions = []
    
    for pii_type, pattern in patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            original = match.group()
            redacted = redacted.replace(original, f'[REDACTED_{pii_type}]')
            redactions.append({
                'type': pii_type,
                'original': original,
                'position': match.span()
            })
    
    return redacted, redactions

def redact_pii_ner(text):
    """Redact PII using Named Entity Recognition"""
    
    doc = nlp(text)
    redacted = text
    redactions = []
    
    # Sort entities by position (reverse) to maintain indices
    entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
    
    for ent in entities:
        # Redact PERSON names and DATE (potential DOB)
        if ent.label_ in ['PERSON', 'DATE']:
            redacted = redacted[:ent.start_char] + f'[REDACTED_{ent.label_}]' + redacted[ent.end_char:]
            redactions.append({
                'type': ent.label_,
                'original': ent.text,
                'position': (ent.start_char, ent.end_char)
            })
    
    return redacted, redactions

def transcribe_and_redact(audio_path):
    """Transcribe audio and redact PII"""
    
    # Step 1: Transcribe
    client = speech.SpeechClient()
    
    with open(audio_path, 'rb') as audio_file:
        content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=16000,
        language_code="en-US",
    )
    
    response = client.recognize(config=config, audio=audio)
    transcript = response.results[0].alternatives[0].transcript
    
    print(f"{'='*60}")
    print(f"ORIGINAL TRANSCRIPT:")
    print(f"{'='*60}")
    print(transcript)
    
    # Step 2: Redact PII using regex
    redacted_regex, redactions_regex = redact_pii_regex(transcript)
    
    # Step 3: Redact PII using NER
    redacted_final, redactions_ner = redact_pii_ner(redacted_regex)
    
    print(f"\n{'='*60}")
    print(f"REDACTED TRANSCRIPT:")
    print(f"{'='*60}")
    print(redacted_final)
    
    # Summary of redactions
    total_redactions = len(redactions_regex) + len(redactions_ner)
    
    if total_redactions > 0:
        print(f"\n{'='*60}")
        print(f"REDACTION SUMMARY:")
        print(f"{'='*60}")
        print(f"Total items redacted: {total_redactions}")
        
        if redactions_regex:
            print(f"\nRegex-based redactions:")
            for r in redactions_regex:
                print(f"  • {r['type']}: {r['original']} → [REDACTED_{r['type']}]")
        
        if redactions_ner:
            print(f"\nNER-based redactions:")
            for r in redactions_ner:
                print(f"  • {r['type']}: {r['original']} → [REDACTED_{r['type']}]")
    else:
        print("\n✅ No PII detected.")
    
    return redacted_final, total_redactions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 3_pii_redaction.py <audio_file>")
        sys.exit(1)
    
    transcribe_and_redact(sys.argv[1])
