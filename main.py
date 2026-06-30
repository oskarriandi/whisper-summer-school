#!/usr/bin/env python3
"""
Whisper Summer School - Medical Transcription Pipeline
Whisper (STT) + BioMistral/OpenMed (SOAP Note Generator)
"""

import os
import sys
import argparse
from transcribe import transcribe_audio
from soap_generator import generate_soap_note, save_soap_note


def run_pipeline(audio_path: str, whisper_model: str = "base", llm_model: str = "llama3.2:3b", language: str = "id"):
    """
    Pipeline lengkap: Audio → Transkrip → SOAP Note

    Args:
        audio_path: Path ke file audio
        whisper_model: tiny | base | small | medium | large
        llm_model: any model available via `ollama list`
        language: kode bahasa ("id", "en") atau "auto" untuk deteksi otomatis
    """
    print("\n" + "="*55)
    print("  🏥 WHISPER SUMMER SCHOOL - MEDICAL PIPELINE")
    print("="*55)
    print(f"  📁 Audio     : {audio_path}")
    print(f"  🎙️  Whisper   : {whisper_model}")
    print(f"  🌐 Language  : {language}")
    print(f"  🤖 LLM Model : {llm_model}")
    print("="*55 + "\n")

    # ── STEP 1: Transkripsi Audio ──────────────────────────
    print("📍 STEP 1: Transkripsi Audio dengan Whisper...")
    transcription = transcribe_audio(audio_path, model_size=whisper_model, language=language)
    
    transcript_text = transcription["text"]
    print(f"\n📝 Hasil Transkripsi ({len(transcript_text)} karakter):")
    print("-" * 40)
    print(transcript_text)
    print("-" * 40)

    # Simpan transkrip
    os.makedirs("outputs", exist_ok=True)
    transcript_file = "outputs/transcript_latest.txt"
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    print(f"💾 Transkrip disimpan: {transcript_file}")

    # ── STEP 2: Generate SOAP Note ─────────────────────────
    print("\n📍 STEP 2: Generate SOAP Note dengan BioMistral...")
    soap_note = generate_soap_note(transcript_text, model=llm_model)

    # ── STEP 3: Simpan Output ──────────────────────────────
    print("\n📍 STEP 3: Menyimpan Output...")
    output_file = save_soap_note(soap_note)

    print("\n" + "="*55)
    print("  ✅ PIPELINE SELESAI!")
    print(f"  📄 Output: {output_file}")
    print("="*55 + "\n")

    return {
        "transcript": transcript_text,
        "soap_note": soap_note,
        "output_file": output_file
    }


def main():
    parser = argparse.ArgumentParser(
        description="Medical Audio Transcription + SOAP Note Generator"
    )
    parser.add_argument(
        "audio", 
        help="Path ke file audio (mp3, wav, m4a, dll)"
    )
    parser.add_argument(
        "--whisper-model", 
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Ukuran model Whisper (default: base)"
    )
    parser.add_argument(
        "--llm-model",
        default="llama3.2:3b",
        help="Model LLM untuk SOAP note (default: llama3.2:3b)"
    )
    parser.add_argument(
        "--language",
        default="id",
        choices=["id", "en", "auto"],
        help="Bahasa audio: id | en | auto (default: id)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"❌ File tidak ditemukan: {args.audio}")
        sys.exit(1)

    result = run_pipeline(
        audio_path=args.audio,
        whisper_model=args.whisper_model,
        llm_model=args.llm_model,
        language=args.language,
    )


if __name__ == "__main__":
    main()
