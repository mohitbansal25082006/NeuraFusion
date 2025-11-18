

# 🧠 NeuraFusion - Multimodal AI Assistant

<div align="center">

![NeuraFusion Logo](https://img.shields.io/badge/NeuraFusion-AI%20Assistant-blue?style=for-the-badge&logo=python)

[![Python](https://img.shields.io/badge/Python-3.10+-green?style=flat-square&logo=python)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.x-orange?style=flat-square&logo=gradio)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

[![Demo](https://img.shields.io/badge/🧪-Try%20Demo-red?style=for-the-badge)](https://huggingface.co/spaces/mohitbansal25082006/NeuraFusion)
[![GitHub](https://img.shields.io/badge/📁-View%20on%20GitHub-black?style=for-the-badge)](https://github.com/mohitbansal25082006/NeuraFusion)

*A powerful multimodal AI assistant that combines text, image, and audio processing with customizable personalities*

</div>

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [🛠️ Technology Stack](#️-technology-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Installation](#-installation)
- [💻 Usage Guide](#-usage-guide)
- [🌐 Deployment Options](#-deployment-options)
- [💰 Cost Analysis](#-cost-analysis)
- [🎓 Learning Outcomes](#-learning-outcomes)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🎯 Project Overview

**NeuraFusion** is a comprehensive multimodal AI assistant that processes text, images, and audio through an intuitive web interface. Built with free-tier Hugging Face models and optional OpenAI integration, it offers a complete AI experience without expensive infrastructure requirements.

### Key Features

- 🧠 **Multimodal Processing**: Understand and process text, images, and audio
- 🎭 **5 AI Personalities**: Choose from Mentor, Friend, Analyst, Professional, or Creative modes
- 🔍 **Visual Analysis**: Advanced image understanding with attention heatmaps
- 🎤 **Voice Interaction**: Speech-to-text and text-to-speech capabilities
- 💾 **Conversation Memory**: Context-aware responses with conversation history
- 📊 **Analytics Dashboard**: Track usage patterns and preferences
- 💼 **Export Options**: Save conversations in multiple formats (JSON, Text, Markdown, CSV)

---

## ✨ Features

### 🧠 Multimodal Capabilities

| Feature | Description | Models Used |
|---------|-------------|-------------|
| **Text Processing** | Advanced text understanding and generation | Flan-T5, GPT-4o (optional) |
| **Image Analysis** | Visual question answering and captioning | BLIP-2, GPT-4o Vision (optional) |
| **Audio Processing** | Speech recognition and synthesis | Whisper, gTTS |
| **Multimodal Fusion** | Combined reasoning across all modalities | Custom fusion engine |

### 🎭 Personality System

1. **🎓 Mentor Mode**
   - Patient and educational approach
   - Detailed explanations with examples
   - Perfect for learning new concepts

2. **😊 Friend Mode** (Default)
   - Casual and conversational tone
   - Warm and empathetic responses
   - Great for everyday chat

3. **📊 Analyst Mode**
   - Data-driven and logical responses
   - Structured analysis approach
   - Ideal for research and analysis

4. **💼 Professional Mode**
   - Formal business communication style
   - Concise and actionable responses
   - Best for work-related queries

5. **🎨 Creative Mode**
   - Imaginative and expressive responses
   - Unique perspectives and ideas
   - Perfect for brainstorming

### 🔍 Visual Analysis Features

- **Attention Heatmaps**: Visualize where the AI "looks" in images
- **Color Distribution Analysis**: RGB channel breakdowns and histograms
- **Feature Visualization**: Extract and display image characteristics
- **Object Recognition**: Identify and describe objects in images

### 💾 Export & Memory System

- **4 Export Formats**: JSON, Text, Markdown, CSV
- **Conversation History**: Complete session tracking with timestamps
- **Context Management**: Intelligent conversation memory
- **Session Analytics**: Usage statistics and preferences

---

## 🛠️ Technology Stack

| Component | Technology | Parameters | Cost |
|-----------|-----------|------------|------|
| **Text Generation** | Flan-T5 Base | 250M | Free |
| **Vision** | BLIP-2 OPT | 2.7B | Free |
| **Speech-to-Text** | Whisper Base | 74M | Free |
| **Text-to-Speech** | gTTS | - | Free |
| **Premium Text** | GPT-4o (optional) | 1.76T | $0.0025/1K tokens |
| **Premium Vision** | GPT-4o Vision | - | $0.01/1K tokens |
| **UI Framework** | Gradio 5.x | - | Free |
| **Visualization** | Matplotlib + Seaborn | - | Free |
| **Memory** | LangChain | - | Free |

---

## 📁 Project Structure

```
F:/NeuraFusion/
│
├── 📁 venv/                          # Anaconda virtual environment
│
├── 📁 models_cache/                  # Downloaded model weights (auto-created)
│
├── 📁 utils/
│   ├── __init__.py
│   ├── text_processor.py             # Text processing utilities
│   ├── image_processor.py            # Image analysis functions
│   ├── audio_processor.py            # Audio transcription & TTS
│   ├── fusion_engine.py              # Multimodal fusion logic
│   ├── memory_manager.py             # Conversation context
│   └── visualization.py              # Attention heatmaps
│
├── 📁 assets/
│   ├── icons/                        # UI icons
│   ├── samples/                      # Sample test files
│   │   ├── sample_image.jpg
│   │   └── sample_audio.mp3
│   └── styles.css                    # Custom CSS for Gradio
│
├── 📁 config/
│   ├── model_configs.json            # Model paths and settings
│   └── personalities.json            # Personality presets
│
├── app.py                            # Main Gradio application
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variables template
├── .gitignore                        # Git ignore file
├── README.md                         # Project documentation
└── test_models.py                    # Script to test model loading
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.10+**
- **Anaconda/Miniconda** (recommended)
- **Git**
- **8GB+ RAM** (16GB recommended)
- **10GB+ free disk space**

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/neurafusion.git
   cd neurafusion
   ```

2. **Create and activate virtual environment**
   ```bash
   conda create -n neurafusion python=3.10
   conda activate neurafusion
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys (optional)
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Open your browser and go to `http://127.0.0.1:7860`

---

## 💻 Usage Guide

### 💬 Smart Chat Tab

**Best For:**
- General conversation
- Q&A
- Learning and explanations
- Brainstorming

**Tips:**
- Switch personalities to match your needs
- Mentor for learning, Analyst for research
- Friend for casual chat, Professional for work

### 🎤 Voice Assistant Tab

**Best For:**
- Hands-free interaction
- Language practice
- Accessibility
- Quick questions

**Tips:**
- Speak clearly in quiet environment
- Keep recordings under 30 seconds
- Use microphone button for live recording
- Upload longer audio files if needed

### 🖼️ Vision Analysis Tab

**Best For:**
- Image description
- Visual question answering
- Color analysis
- Object identification

**Features to Try:**
- Ask specific questions about images
- Enable attention maps to see AI focus
- Use color analysis for design insights
- Try different personalities for varied descriptions

### 🔗 Complete Fusion Tab

**Best For:**
- Complex queries with multiple inputs
- Research and analysis
- Content creation
- Comprehensive understanding

**Examples:**
- Upload vacation photo + ask "Where was this taken?" via voice
- Show recipe image + ask "How can I make this healthier?" in text
- Record audio question + upload relevant image for context

### 📊 Analytics Tab

**Best For:**
- Tracking usage patterns
- Understanding your habits
- System monitoring
- Performance insights

### 💾 Memory & Export Tab

**Best For:**
- Saving conversations
- Reviewing history
- Sharing insights
- Data analysis

---

## 🌐 Deployment Options

### Option 1: Local Development

- Runs on your PC
- Access via: http://127.0.0.1:7860
- Private and secure

### Option 2: Hugging Face Spaces (Free Hosting)

1. Create a Hugging Face account at https://huggingface.co/join
2. Create a new Space with the Gradio SDK
3. Upload all project files
4. Add your environment variables
5. Deploy and get a public URL

### Option 3: Share Temporarily

- In `app.py`, change: `share=True`
- Get a temporary public link
- Valid for 72 hours
- No setup needed!

---

## 💰 Cost Analysis

| Component | Cost |
|-----------|------|
| Hugging Face Models | $0 (free) |
| HF Spaces Hosting | $0 (free tier) |
| gTTS (Text-to-Speech) | $0 (free) |
| OpenAI GPT-4o API (optional) | $5-20 (pay-as-you-go) |
| ElevenLabs TTS (optional) | $0-5 (free tier: 10k chars/month) |
| **Total (Free Version)** | **$0** |
| **Total (Premium Version)** | **$5-25** |

---

## 🎓 Learning Outcomes

By completing this project, you'll master:

✅ **Hugging Face Ecosystem** - Transformers, Spaces, Models Hub  
✅ **Multimodal AI** - Text, Vision, Audio processing  
✅ **Gradio Framework** - Interactive ML demos  
✅ **LangChain** - AI orchestration and memory  
✅ **API Integration** - OpenAI, Hugging Face, ElevenLabs  
✅ **Deployment** - Cloud hosting, Git workflows  
✅ **Python Best Practices** - Virtual environments, project structure  

---

## 🤝 Contributing

We welcome contributions to NeuraFusion! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow the existing code style and conventions
- Write clear, descriptive commit messages
- Add tests for new features
- Update documentation as needed

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### AI Models
- **Flan-T5**: Google Research
- **BLIP-2**: Salesforce Research
- **Whisper**: OpenAI
- **GPT-4o**: OpenAI

### Frameworks
- **Gradio**: Hugging Face Team
- **Transformers**: Hugging Face
- **LangChain**: LangChain Team

### Tools
- **Python**: PSF
- **PyTorch**: Meta AI
- **Matplotlib**: NumFOCUS

---

<div align="center">

**Made with ❤️ and 🧠 | NeuraFusion V3.0**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/neurafusion?style=social)](https://github.com/mohitbansal25082006/neurafusion)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/neurafusion?style=social)](https://github.com/mohitbansal25082006/neurafusion)

</div>
