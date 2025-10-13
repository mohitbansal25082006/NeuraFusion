"""
Audio Processing Module - Handles speech-to-text and text-to-speech
Part 2: NeuraFusion Audio Integration
"""

import whisper
import torch
from gtts import gTTS
import tempfile
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class AudioProcessor:
    """
    Manages audio processing using Whisper (STT) and gTTS (TTS).
    
    Features:
    - Speech-to-Text: Convert voice recordings to text using OpenAI Whisper
    - Text-to-Speech: Convert AI responses to audio using Google TTS
    - Audio file handling and preprocessing
    """
    
    def __init__(self, whisper_model="base", cache_dir="./models_cache"):
        """
        Initialize the audio processor.
        
        Args:
            whisper_model: Whisper model size
                - 'tiny' = 39M params, fastest, least accurate
                - 'base' = 74M params, good balance (RECOMMENDED)
                - 'small' = 244M params, better accuracy, slower
                - 'medium' = 769M params, best accuracy, very slow
            cache_dir: Directory to cache downloaded models
        """
        print(f"🔄 Loading Whisper model: {whisper_model}")
        print("⏳ First-time download may take 1-2 minutes...")
        
        # Set cache directory for Whisper
        os.environ['WHISPER_CACHE_DIR'] = cache_dir
        
        # Load Whisper model
        try:
            self.whisper_model = whisper.load_model(
                whisper_model,
                download_root=cache_dir
            )
            print("✅ Whisper (Speech-to-Text) loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Whisper: {e}")
            print("💡 Trying to download model again...")
            self.whisper_model = whisper.load_model(whisper_model)
        
        # TTS settings
        self.tts_language = "en"  # Default language for text-to-speech
        self.tts_speed = False    # Normal speed (True = slower)
        
        print("✅ Audio processor initialized!")
    
    def transcribe_audio(self, audio_path, language="en"):
        """
        Convert speech to text using Whisper.
        
        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            language: Language code (en, es, fr, etc.)
                     If None, Whisper auto-detects language
        
        Returns:
            Dictionary with:
                - 'text': Transcribed text
                - 'language': Detected language
                - 'confidence': Average word confidence (if available)
        """
        
        if not audio_path or not os.path.exists(audio_path):
            return {
                'text': "",
                'language': "unknown",
                'error': "Audio file not found"
            }
        
        print(f"🎤 Transcribing audio: {Path(audio_path).name}")
        
        try:
            # Transcribe using Whisper
            # fp16=False ensures CPU compatibility
            result = self.whisper_model.transcribe(
                audio_path,
                language=language if language != "auto" else None,
                fp16=False,  # Use float32 for CPU
                verbose=False  # Don't print progress
            )
            
            transcribed_text = result['text'].strip()
            detected_language = result.get('language', language)
            
            print(f"✅ Transcription complete: '{transcribed_text[:50]}...'")
            
            return {
                'text': transcribed_text,
                'language': detected_language,
                'segments': result.get('segments', []),
                'error': None
            }
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return {
                'text': "",
                'language': "unknown",
                'error': str(e)
            }
    
    def text_to_speech(self, text, output_path=None, language="en", slow=False):
        """
        Convert text to speech using Google TTS.
        
        Args:
            text: Text to convert to speech
            output_path: Where to save the audio file (optional)
                        If None, creates temp file
            language: Language code (en, es, fr, etc.)
            slow: If True, speaks slower
        
        Returns:
            Path to generated audio file
        """
        
        if not text or len(text.strip()) == 0:
            print("⚠️ No text provided for TTS")
            return None
        
        print(f"🔊 Generating speech: '{text[:50]}...'")
        
        try:
            # Create gTTS object
            tts = gTTS(
                text=text,
                lang=language,
                slow=slow
            )
            
            # Determine output path
            if output_path is None:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp3"
                )
                output_path = temp_file.name
                temp_file.close()
            
            # Save audio file
            tts.save(output_path)
            
            print(f"✅ Speech generated: {Path(output_path).name}")
            return output_path
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return None
    
    def process_voice_input(self, audio_file):
        """
        Gradio-compatible function for voice input.
        Handles the full pipeline: audio -> text
        
        Args:
            audio_file: Path to uploaded audio file (from Gradio)
        
        Returns:
            Transcribed text string
        """
        
        if audio_file is None:
            return ""
        
        result = self.transcribe_audio(audio_file)
        
        if result['error']:
            return f"[Transcription Error: {result['error']}]"
        
        return result['text']
    
    def generate_voice_response(self, text, language="en"):
        """
        Gradio-compatible function for voice output.
        Handles the full pipeline: text -> audio file
        
        Args:
            text: Text to convert to speech
            language: Language code
        
        Returns:
            Path to generated audio file (for Gradio Audio component)
        """
        
        if not text or len(text.strip()) == 0:
            return None
        
        # Generate speech
        audio_path = self.text_to_speech(text, language=language)
        
        return audio_path
    
    def batch_transcribe(self, audio_files, language="en"):
        """
        Transcribe multiple audio files at once.
        
        Args:
            audio_files: List of audio file paths
            language: Language code for all files
        
        Returns:
            List of transcription results
        """
        results = []
        
        for audio_file in audio_files:
            result = self.transcribe_audio(audio_file, language)
            results.append(result)
        
        return results
    
    def get_audio_info(self, audio_path):
        """
        Get metadata about an audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary with audio information
        """
        try:
            import soundfile as sf
            
            data, samplerate = sf.read(audio_path)
            duration = len(data) / samplerate
            
            return {
                'duration': duration,
                'sample_rate': samplerate,
                'channels': len(data.shape) if len(data.shape) > 1 else 1,
                'samples': len(data)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def supported_languages(self):
        """
        Get list of supported languages for TTS.
        
        Returns:
            Dictionary of language codes and names
        """
        return {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh-CN': 'Chinese (Simplified)',
            'hi': 'Hindi',
            'ar': 'Arabic'
        }


# Test function
if __name__ == "__main__":
    print("🧪 Testing Audio Processor...")
    print("="*60)
    
    processor = AudioProcessor(whisper_model="base")
    
    print("\n" + "="*60)
    print("📋 Test 1: Text-to-Speech")
    print("="*60)
    
    test_text = "Hello! This is NeuraFusion, your multimodal AI assistant. I can understand text, images, and now audio too!"
    
    audio_file = processor.text_to_speech(test_text, output_path="test_tts.mp3")
    
    if audio_file:
        print(f"✅ Audio generated: {audio_file}")
        print(f"📁 File size: {os.path.getsize(audio_file)} bytes")
        
        # Get audio info
        info = processor.get_audio_info(audio_file)
        if 'duration' in info:
            print(f"⏱️  Duration: {info['duration']:.2f} seconds")
    else:
        print("❌ TTS failed")
    
    print("\n" + "="*60)
    print("📋 Test 2: Supported Languages")
    print("="*60)
    
    languages = processor.supported_languages()
    print(f"🌍 Supported languages: {len(languages)}")
    for code, name in list(languages.items())[:5]:
        print(f"  - {code}: {name}")
    print("  ... and more!")
    
    print("\n" + "="*60)
    print("✅ Audio Processor tests complete!")
    print("="*60)
    print("\n💡 To test Speech-to-Text, record an audio file and run:")
    print("   result = processor.transcribe_audio('your_audio.mp3')")
    print("   print(result['text'])")