"""
Deploy NeuraFusion to Hugging Face Spaces
Minimal deployment script - creates ready-to-upload package
"""

import os
import shutil
from pathlib import Path

def create_deployment_package():
    """Create minimal deployment package for Hugging Face Spaces."""
    
    print("🚀 Creating NeuraFusion deployment package...")
    
    # Create deployment directory
    deploy_dir = Path("deploy_package")
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()
    
    # Essential files to copy
    essential_files = [
        'app.py',
        'requirements.txt', 
        '.gitignore'
    ]
    
    # Essential directories to copy
    essential_dirs = [
        'utils',
        'config',
        'assets'
    ]
    
    # Copy files
    print("📄 Copying files...")
    for file in essential_files:
        if Path(file).exists():
            shutil.copy(file, deploy_dir / file)
            print(f"  ✓ {file}")
    
    # Copy directories
    print("📁 Copying directories...")
    for directory in essential_dirs:
        if Path(directory).exists():
            shutil.copytree(directory, deploy_dir / directory)
            print(f"  ✓ {directory}/")
    
    # Create Spaces-specific files
    create_spaces_readme(deploy_dir)
    
    print(f"✅ Deployment package created: {deploy_dir}/")
    return deploy_dir

def create_spaces_readme(deploy_dir):
    """Create minimal README for Spaces."""
    
    readme_content = """---
title: NeuraFusion
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# NeuraFusion

Multimodal AI Assistant with text, image, and audio understanding.

## Features

- 💬 Smart Chat with personality modes
- 🎤 Voice Assistant (Speech-to-Text & Text-to-Speech) 
- 🖼️ Image Analysis with BLIP-2
- 🔗 Multimodal Fusion
- 📊 Analytics & Export

## Usage

1. Select a personality mode
2. Use any input method: text, voice, or image
3. Get intelligent multimodal responses

All features work without API keys - 100% free!
"""
    
    with open(deploy_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("  ✓ README.md")

def show_upload_instructions(deploy_dir):
    """Show minimal upload instructions."""
    
    print("\n📤 UPLOAD INSTRUCTIONS:")
    print("1. Go to: https://huggingface.co/new-space")
    print("2. Create Space with:")
    print("   - Owner: your_username")
    print("   - Space name: neurafusion") 
    print("   - SDK: Gradio")
    print("   - Hardware: CPU Basic (free)")
    print("3. Upload ALL files from deploy_package/")
    print("4. Wait 5-10 minutes for build")
    print("5. Your Space will be live at:")
    print("   https://huggingface.co/spaces/your_username/neurafusion")
    print(f"\n📁 Upload these files from: {deploy_dir.absolute()}")

def main():
    """Main deployment function."""
    
    try:
        # Create deployment package
        deploy_dir = create_deployment_package()
        
        # Show instructions
        show_upload_instructions(deploy_dir)
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")

if __name__ == "__main__":
    main()