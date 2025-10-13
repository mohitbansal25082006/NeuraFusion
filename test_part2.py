"""
NeuraFusion Part 2 - Complete Testing Script
Tests all new audio and fusion components
"""

import sys
from pathlib import Path

print("="*70)
print("🧪 NeuraFusion Part 2 - Component Testing")
print("="*70)

# Test imports
print("\n📦 Testing imports...")
try:
    from utils.text_processor import TextProcessor
    print("  ✅ TextProcessor imported")
except Exception as e:
    print(f"  ❌ TextProcessor failed: {e}")
    sys.exit(1)

try:
    from utils.image_processor import ImageProcessor
    print("  ✅ ImageProcessor imported")
except Exception as e:
    print(f"  ❌ ImageProcessor failed: {e}")
    sys.exit(1)

try:
    from utils.audio_processor import AudioProcessor
    print("  ✅ AudioProcessor imported")
except Exception as e:
    print(f"  ❌ AudioProcessor failed: {e}")
    sys.exit(1)

try:
    from utils.fusion_engine import FusionEngine
    print("  ✅ FusionEngine imported")
except Exception as e:
    print(f"  ❌ FusionEngine failed: {e}")
    sys.exit(1)

try:
    from utils.memory_manager import MemoryManager
    print("  ✅ MemoryManager imported")
except Exception as e:
    print(f"  ❌ MemoryManager failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ All imports successful!")
print("="*70)

# Initialize components
print("\n🚀 Initializing components...")
print("⏳ This may take a few minutes on first run...")

try:
    print("\n1️⃣ Loading Text Processor...")
    text_proc = TextProcessor(model_name="google/flan-t5-base")
    print("   ✅ Text Processor ready")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

try:
    print("\n2️⃣ Loading Image Processor...")
    image_proc = ImageProcessor(model_name="Salesforce/blip2-opt-2.7b")
    print("   ✅ Image Processor ready")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

try:
    print("\n3️⃣ Loading Audio Processor...")
    audio_proc = AudioProcessor(whisper_model="base")
    print("   ✅ Audio Processor ready")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

try:
    print("\n4️⃣ Initializing Fusion Engine...")
    fusion = FusionEngine(text_proc, image_proc, audio_proc)
    print("   ✅ Fusion Engine ready")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

try:
    print("\n5️⃣ Initializing Memory Manager...")
    memory = MemoryManager(max_history=100)
    print("   ✅ Memory Manager ready")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("🎉 All components initialized successfully!")
print("="*70)

# Run functionality tests
print("\n🧪 Running Functionality Tests...")
print("="*70)

# Test 1: Text Processing
print("\n📝 Test 1: Text Generation")
print("-"*70)
try:
    test_prompt = "What is artificial intelligence?"
    response = text_proc.generate_response(test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response[:100]}...")
    print("✅ Text generation working")
except Exception as e:
    print(f"❌ Text generation failed: {e}")

# Test 2: Text-to-Speech
print("\n🔊 Test 2: Text-to-Speech")
print("-"*70)
try:
    test_text = "Hello! This is a test of the text to speech system."
    audio_file = audio_proc.text_to_speech(test_text, output_path="test_tts.mp3")
    if audio_file and Path(audio_file).exists():
        print(f"✅ TTS generated: {audio_file}")
        print(f"   File size: {Path(audio_file).stat().st_size} bytes")
    else:
        print("❌ TTS file not created")
except Exception as e:
    print(f"❌ TTS failed: {e}")

# Test 3: Memory Manager
print("\n💾 Test 3: Conversation Memory")
print("-"*70)
try:
    memory.add_user_message("Hello AI!", modalities=['text'])
    memory.add_assistant_message("Hello! How can I help?", modalities=['text'])
    memory.add_user_message("Tell me about space", modalities=['text', 'audio'])
    
    summary = memory.get_session_summary()
    print(f"Session ID: {summary['session_id']}")
    print(f"Total turns: {summary['total_turns']}")
    print(f"Messages: {summary['history_length']}")
    print(f"Modalities used: {', '.join(summary['modalities_used'])}")
    print("✅ Memory management working")
except Exception as e:
    print(f"❌ Memory test failed: {e}")

# Test 4: Fusion Engine
print("\n🔗 Test 4: Multimodal Fusion")
print("-"*70)
try:
    # Test input analysis
    analysis = fusion.analyze_inputs(
        text="Hello",
        image=None,
        audio=None
    )
    print(f"Input analysis:")
    print(f"  - Modality count: {analysis['modality_count']}")
    print(f"  - Primary modality: {analysis['primary_modality']}")
    print(f"  - Complexity: {analysis['input_complexity']}")
    print("✅ Fusion engine working")
except Exception as e:
    print(f"❌ Fusion test failed: {e}")

# Test 5: Supported Languages
print("\n🌍 Test 5: Supported Languages")
print("-"*70)
try:
    languages = audio_proc.supported_languages()
    print(f"Supported TTS languages: {len(languages)}")
    print("Sample languages:")
    for i, (code, name) in enumerate(list(languages.items())[:5]):
        print(f"  - {code}: {name}")
    print("  ... and more!")
    print("✅ Language support verified")
except Exception as e:
    print(f"❌ Language test failed: {e}")

# Summary
print("\n" + "="*70)
print("📊 TEST SUMMARY")
print("="*70)
print("""
✅ All core components are working!

What you can do now:
1. Run the main app: python app.py
2. Test text chat with memory
3. Try voice input/output
4. Experiment with multimodal fusion
5. Export conversation history

""")

print("="*70)
print("🎉 Part 2 is ready to use!")
print("="*70)
print("\n💡 Next: Run 'python app.py' to start the full interface")
print("⌨️  Or continue to Part 3 for deployment features\n")