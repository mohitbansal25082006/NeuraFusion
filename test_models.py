"""
Model Testing Script
Run this before launching the main app to verify everything works
"""

import sys
from pathlib import Path

print("="*60)
print("🧪 NeuraFusion Model Testing Script")
print("="*60)

# Test 1: Check Python version
print("\n📌 Test 1: Python Version")
print(f"   Python {sys.version}")
if sys.version_info < (3, 10):
    print("   ⚠️  Warning: Python 3.10+ recommended")
else:
    print("   ✅ Python version OK")

# Test 2: Check required packages
print("\n📌 Test 2: Package Imports")
required_packages = {
    "torch": "PyTorch",
    "transformers": "Hugging Face Transformers",
    "gradio": "Gradio UI Framework",
    "PIL": "Pillow (Image Processing)"
}

for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"   ✅ {name} installed")
    except ImportError:
        print(f"   ❌ {name} NOT installed")
        print(f"      Install with: pip install {package}")

# Test 3: Check folder structure
print("\n📌 Test 3: Folder Structure")
required_folders = ["utils", "config", "assets", "models_cache"]
for folder in required_folders:
    if Path(folder).exists():
        print(f"   ✅ {folder}/ exists")
    else:
        print(f"   ❌ {folder}/ missing - creating it...")
        Path(folder).mkdir(exist_ok=True)

# Test 4: Load Text Model
print("\n📌 Test 4: Text Model (Flan-T5)")
print("   Loading... (first time may take 2-3 minutes)")
try:
    from utils.text_processor import TextProcessor
    text_proc = TextProcessor(cache_dir="./models_cache")
    test_response = text_proc.generate_response("Hello, how are you?")
    print(f"   ✅ Text model loaded!")
    print(f"   🤖 Test response: {test_response[:100]}...")
except Exception as e:
    print(f"   ❌ Error loading text model: {str(e)}")

# Test 5: Load Image Model
print("\n📌 Test 5: Image Model (BLIP-2)")
print("   Loading... (first time may take 5-7 minutes)")
try:
    from utils.image_processor import ImageProcessor
    from PIL import Image
    
    img_proc = ImageProcessor(cache_dir="./models_cache")
    
    # Create a simple test image
    test_img = Image.new('RGB', (224, 224), color='blue')
    caption = img_proc.caption_image(test_img)
    print(f"   ✅ Image model loaded!")
    print(f"   🖼️  Test caption: {caption}")
except Exception as e:
    print(f"   ❌ Error loading image model: {str(e)}")

# Final Summary
print("\n" + "="*60)
print("📊 Test Summary")
print("="*60)
print("\n✅ If all tests passed, you can run: python app.py")
print("❌ If any tests failed, check the error messages above")
print("\n💡 Tip: First-time model downloads require internet connection")
print("="*60)