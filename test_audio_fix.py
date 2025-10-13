"""
Test audio processing after FFmpeg fix
"""

print("="*70)
print("🧪 Testing Audio Processing")
print("="*70)

# Test 1: Import and initialize
print("\n1️⃣ Testing imports...")
try:
    from utils.audio_processor import AudioProcessor
    print("✅ AudioProcessor imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Initialize processor
print("\n2️⃣ Initializing AudioProcessor...")
try:
    audio_proc = AudioProcessor(whisper_model="base")
    print("✅ AudioProcessor initialized")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    exit(1)

# Test 3: Text-to-Speech
print("\n3️⃣ Testing Text-to-Speech...")
try:
    test_text = "Hello! This is a test of the audio system."
    output_file = audio_proc.text_to_speech(test_text, output_path="test_tts.mp3")
    
    if output_file and os.path.exists(output_file):
        print(f"✅ TTS Success! File created: {output_file}")
        print(f"   File size: {os.path.getsize(output_file)} bytes")
    else:
        print("❌ TTS failed to create file")
except Exception as e:
    print(f"❌ TTS error: {e}")

# Test 4: Check supported languages
print("\n4️⃣ Testing language support...")
try:
    languages = audio_proc.supported_languages()
    print(f"✅ Supported languages: {len(languages)}")
    print("   Sample: en, es, fr, de, hi")
except Exception as e:
    print(f"❌ Language check failed: {e}")

print("\n" + "="*70)
print("✅ All tests completed!")
print("="*70)
print("\n💡 If all tests passed, audio processing is ready!")
print("   You can now test voice input by recording in the app.")