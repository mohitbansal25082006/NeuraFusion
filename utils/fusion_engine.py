"""
Multimodal Fusion Engine - Combines text, image, and audio inputs
Part 2: NeuraFusion Intelligent Input Fusion
"""

from datetime import datetime
import json

class FusionEngine:
    """
    Intelligent fusion of multiple input modalities.
    
    This engine:
    1. Analyzes which inputs are provided (text, image, audio)
    2. Processes each modality appropriately
    3. Combines results into coherent responses
    4. Maintains context across interactions
    5. Supports personality integration and OpenAI enhancement
    """
    
    def __init__(self, text_processor, image_processor, audio_processor):
        """
        Initialize fusion engine with all processors.
        
        Args:
            text_processor: TextProcessor instance
            image_processor: ImageProcessor instance
            audio_processor: AudioProcessor instance
        """
        self.text_proc = text_processor
        self.image_proc = image_processor
        self.audio_proc = audio_processor
        
        # Fusion modes
        self.modes = {
            'balanced': 'Give equal weight to all inputs',
            'text_focused': 'Prioritize text information',
            'visual_focused': 'Prioritize image information',
            'audio_focused': 'Prioritize audio information'
        }
        
        self.current_mode = 'balanced'
        
        print("✅ Fusion Engine initialized!")
    
    def analyze_inputs(self, text=None, image=None, audio=None):
        """
        Analyze which inputs are provided and their properties.
        
        Args:
            text: Text string
            image: PIL Image or path
            audio: Audio file path
        
        Returns:
            Dictionary with input analysis
        """
        analysis = {
            'has_text': bool(text and len(str(text).strip()) > 0),
            'has_image': image is not None,
            'has_audio': bool(audio and audio.strip()) if isinstance(audio, str) else False,
            'modality_count': 0,
            'primary_modality': None,
            'input_complexity': 'simple'
        }
        
        # Count active modalities
        active = []
        if analysis['has_text']:
            active.append('text')
        if analysis['has_image']:
            active.append('image')
        if analysis['has_audio']:
            active.append('audio')
        
        analysis['modality_count'] = len(active)
        analysis['active_modalities'] = active
        
        # Determine primary modality
        if analysis['modality_count'] == 0:
            analysis['primary_modality'] = None
        elif analysis['modality_count'] == 1:
            analysis['primary_modality'] = active[0]
            analysis['input_complexity'] = 'simple'
        else:
            analysis['primary_modality'] = 'multimodal'
            analysis['input_complexity'] = 'complex'
        
        return analysis
    
    def fuse_multimodal_input(self, text=None, image=None, audio=None, 
                              mode='balanced', include_voice_output=False):
        """
        Main fusion function - processes all inputs and generates response.
        
        Args:
            text: Text input string
            image: PIL Image or path
            audio: Audio file path or transcribed text
            mode: Fusion mode ('balanced', 'text_focused', etc.)
            include_voice_output: If True, also generate audio response
        
        Returns:
            Dictionary with:
                - 'text_response': Text answer
                - 'audio_response': Audio file path (if requested)
                - 'metadata': Processing information
        """
        
        # Analyze inputs
        analysis = self.analyze_inputs(text, image, audio)
        
        # Initialize response components
        response_parts = []
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'input_analysis': analysis,
            'processing_steps': []
        }
        
        # ======================================
        # STEP 1: Process Audio (if provided)
        # ======================================
        audio_text = None
        if analysis['has_audio']:
            metadata['processing_steps'].append('audio_transcription')
            
            # If audio is already transcribed text, use it directly
            if isinstance(audio, str) and not audio.endswith(('.mp3', '.wav', '.m4a')):
                audio_text = audio
            else:
                # Transcribe audio file
                transcription = self.audio_proc.transcribe_audio(audio)
                audio_text = transcription.get('text', '')
                metadata['audio_transcription'] = {
                    'success': bool(audio_text),
                    'language': transcription.get('language', 'unknown')
                }
            
            if audio_text:
                response_parts.append(f"🎤 **Voice Input Detected:**\n*\"{audio_text}\"*\n")
        
        # ======================================
        # STEP 2: Process Image (if provided)
        # ======================================
        image_description = None
        if analysis['has_image']:
            metadata['processing_steps'].append('image_analysis')
            
            # Check if there's a question about the image
            question_text = audio_text or text
            
            if question_text and len(question_text.strip()) > 0:
                # Visual Question Answering
                image_answer = self.image_proc.answer_question(image, question_text)
                image_description = image_answer
                response_parts.append(f"🖼️ **Image Analysis (Q&A):**\n{image_answer}\n")
            else:
                # Just caption the image
                caption = self.image_proc.caption_image(image)
                image_description = caption
                response_parts.append(f"🖼️ **Image Description:**\n{caption}\n")
            
            metadata['image_analysis'] = {
                'method': 'vqa' if question_text else 'captioning',
                'success': bool(image_description)
            }
        
        # ======================================
        # STEP 3: Process Text Query
        # ======================================
        text_response = None
        
        # Combine all text sources
        combined_text = []
        if audio_text:
            combined_text.append(audio_text)
        if text and len(str(text).strip()) > 0:
            combined_text.append(str(text))
        
        query = " ".join(combined_text).strip()
        
        if query or image_description:
            metadata['processing_steps'].append('text_generation')
            
            # Create context-aware prompt
            if image_description and query:
                # User asked about an image
                prompt = f"Based on this image description: '{image_description}', answer the question: {query}"
            elif image_description and not query:
                # Just describe the image more
                prompt = f"Provide additional insights about an image described as: '{image_description}'"
            else:
                # Pure text query
                prompt = query
            
            # Generate response using text processor
            text_response = self.text_proc.generate_response(prompt)
            response_parts.append(f"💬 **Response:**\n{text_response}")
            
            metadata['text_generation'] = {
                'prompt_length': len(prompt),
                'response_length': len(text_response),
                'success': bool(text_response)
            }
        
        # ======================================
        # STEP 4: Combine All Parts
        # ======================================
        if not response_parts:
            final_text = "I didn't receive any input. Please provide text, an image, or audio."
        else:
            final_text = "\n".join(response_parts)
        
        # ======================================
        # STEP 5: Generate Voice Output (optional)
        # ======================================
        audio_output = None
        if include_voice_output and text_response:
            metadata['processing_steps'].append('tts_generation')
            audio_output = self.audio_proc.text_to_speech(text_response)
            metadata['tts_generation'] = {
                'success': audio_output is not None
            }
        
        # ======================================
        # RETURN RESULTS
        # ======================================
        return {
            'text_response': final_text,
            'audio_response': audio_output,
            'metadata': metadata,
            'components': {
                'audio_transcription': audio_text,
                'image_description': image_description,
                'text_answer': text_response
            }
        }
    
    def fuse_with_personality(self, text=None, image=None, audio=None, 
                             personality_manager=None, openai_integration=None,
                             include_voice_output=False):
        """
        Enhanced fusion with personality and optional OpenAI.
        
        Args:
            text: Text input
            image: Image input
            audio: Audio input
            personality_manager: PersonalityManager instance
            openai_integration: OpenAIIntegration instance
            include_voice_output: Generate voice response
        
        Returns:
            Enhanced fusion result dictionary
        """
        
        # Get personality context
        personality_context = None
        if personality_manager:
            personality_context = personality_manager.get_system_prompt()
        
        # Analyze inputs
        analysis = self.analyze_inputs(text, image, audio)
        
        # Initialize response components
        response_parts = []
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'input_analysis': analysis,
            'processing_steps': [],
            'personality': personality_manager.current_personality if personality_manager else 'default',
            'used_openai': False
        }
        
        # Process Audio
        audio_text = None
        if analysis['has_audio']:
            metadata['processing_steps'].append('audio_transcription')
            
            if isinstance(audio, str) and not audio.endswith(('.mp3', '.wav', '.m4a')):
                audio_text = audio
            else:
                transcription = self.audio_proc.transcribe_audio(audio)
                audio_text = transcription.get('text', '')
                metadata['audio_transcription'] = {
                    'success': bool(audio_text),
                    'language': transcription.get('language', 'unknown')
                }
            
            if audio_text:
                response_parts.append(f"🎤 **Voice Input:** *\"{audio_text}\"*\n")
        
        # Process Image
        image_description = None
        if analysis['has_image']:
            metadata['processing_steps'].append('image_analysis')
            
            # Try OpenAI first if available
            if openai_integration and openai_integration.is_enabled():
                question_text = audio_text or text
                image_description = openai_integration.analyze_image(image, question_text)
                metadata['used_openai'] = True
            else:
                # Use local BLIP-2
                question_text = audio_text or text
                if question_text and len(question_text.strip()) > 0:
                    image_description = self.image_proc.answer_question(image, question_text)
                else:
                    image_description = self.image_proc.caption_image(image)
            
            response_parts.append(f"🖼️ **Image Analysis:** {image_description}\n")
            
            metadata['image_analysis'] = {
                'method': 'openai' if metadata['used_openai'] else 'blip2',
                'success': bool(image_description)
            }
        
        # Process Text Query
        text_response = None
        combined_text = []
        if audio_text:
            combined_text.append(audio_text)
        if text and len(str(text).strip()) > 0:
            combined_text.append(str(text))
        
        query = " ".join(combined_text).strip()
        
        if query or image_description:
            metadata['processing_steps'].append('text_generation')
            
            # Build prompt
            if image_description and query:
                prompt = f"Based on this image: '{image_description}', {query}"
            elif image_description:
                prompt = f"Describe insights about: '{image_description}'"
            else:
                prompt = query
            
            # Try OpenAI first if available and user wants it
            if openai_integration and openai_integration.is_enabled():
                personality_config = personality_manager.get_current_personality() if personality_manager else None
                text_response = openai_integration.generate_text(
                    prompt, 
                    personality_config=personality_config
                )
                metadata['used_openai'] = True
            else:
                # Use local Flan-T5 with personality
                text_response = self.text_proc.generate_response(
                    prompt,
                    personality_context=personality_context
                )
            
            response_parts.append(f"💬 **Response:** {text_response}")
            
            metadata['text_generation'] = {
                'prompt_length': len(prompt),
                'response_length': len(text_response),
                'model': 'gpt-4o' if metadata['used_openai'] else 'flan-t5',
                'success': bool(text_response)
            }
        
        # Combine parts
        final_text = "\n".join(response_parts) if response_parts else "No input received."
        
        # Generate voice output
        audio_output = None
        if include_voice_output and text_response:
            metadata['processing_steps'].append('tts_generation')
            audio_output = self.audio_proc.text_to_speech(text_response)
            metadata['tts_generation'] = {'success': audio_output is not None}
        
        return {
            'text_response': final_text,
            'audio_response': audio_output,
            'metadata': metadata,
            'components': {
                'audio_transcription': audio_text,
                'image_description': image_description,
                'text_answer': text_response
            }
        }
    
    def generate_with_personality(self, text, personality_manager, openai_integration=None):
        """
        Generate response with personality mode.
        
        Args:
            text: User input
            personality_manager: PersonalityManager instance
            openai_integration: Optional OpenAI integration
        
        Returns:
            Response text
        """
        if not text or len(text.strip()) == 0:
            return "Please provide a message."
        
        # Try OpenAI if available
        if openai_integration and openai_integration.is_enabled():
            personality_config = personality_manager.get_current_personality()
            return openai_integration.generate_text(text, personality_config)
        
        # Use local with personality
        personality_context = personality_manager.get_system_prompt()
        return self.text_proc.generate_response(
            text,
            personality_context=personality_context,
            temperature=personality_manager.get_temperature(),
            max_length=personality_manager.get_max_length()
        )
    
    def simple_text_response(self, text):
        """
        Quick text-only response (for chat interface).
        
        Args:
            text: User's text message
        
        Returns:
            Bot's text response
        """
        if not text or len(text.strip()) == 0:
            return "Please provide a message."
        
        response = self.text_proc.generate_response(text)
        return response
    
    def simple_image_response(self, image, question=None):
        """
        Quick image-only response (for image interface).
        
        Args:
            image: PIL Image
            question: Optional question about image
        
        Returns:
            Image analysis text
        """
        if image is None:
            return "Please provide an image."
        
        return self.image_proc.analyze_image(image, question)
    
    def simple_audio_response(self, audio_file):
        """
        Quick audio-only response (for voice interface).
        
        Args:
            audio_file: Path to audio file
        
        Returns:
            Transcribed text
        """
        if not audio_file:
            return "Please provide an audio file."
        
        result = self.audio_proc.transcribe_audio(audio_file)
        
        if result['error']:
            return f"Transcription error: {result['error']}"
        
        # Generate response to transcribed text
        transcribed = result['text']
        response = self.text_proc.generate_response(transcribed)
        
        return f"🎤 You said: \"{transcribed}\"\n\n💬 Response: {response}"
    
    def get_processing_summary(self, metadata):
        """
        Generate a human-readable summary of processing steps.
        
        Args:
            metadata: Metadata dictionary from fusion result
        
        Returns:
            Summary string
        """
        steps = metadata.get('processing_steps', [])
        analysis = metadata.get('input_analysis', {})
        
        summary_parts = [
            f"📊 Processing Summary:",
            f"- Input modalities: {analysis.get('modality_count', 0)}",
            f"- Active inputs: {', '.join(analysis.get('active_modalities', []))}",
            f"- Processing steps: {len(steps)}",
            f"- Steps performed: {', '.join(steps)}"
        ]
        
        return "\n".join(summary_parts)
    
    def set_fusion_mode(self, mode):
        """
        Change the fusion mode.
        
        Args:
            mode: One of 'balanced', 'text_focused', 'visual_focused', 'audio_focused'
        """
        if mode in self.modes:
            self.current_mode = mode
            return f"Fusion mode set to: {mode}"
        else:
            return f"Invalid mode. Choose from: {', '.join(self.modes.keys())}"
    
    def get_capabilities(self):
        """
        Get a summary of fusion engine capabilities.
        
        Returns:
            Dictionary describing capabilities
        """
        return {
            'supported_modalities': ['text', 'image', 'audio'],
            'fusion_modes': list(self.modes.keys()),
            'current_mode': self.current_mode,
            'features': [
                'Automatic modality detection',
                'Cross-modal reasoning',
                'Context-aware responses',
                'Voice input/output',
                'Visual question answering',
                'Multi-turn conversation',
                'Personality integration',
                'OpenAI enhancement'
            ]
        }


# Test function
if __name__ == "__main__":
    print("🧪 Testing Fusion Engine...")
    print("="*60)
    
    # This test requires all processors to be initialized
    # Normally you would import and initialize them
    print("⚠️  Full test requires TextProcessor, ImageProcessor, and AudioProcessor")
    print("💡 Demonstrating input analysis only...")
    
    # Mock fusion engine (without actual processors)
    class MockProcessor:
        def generate_response(self, text):
            return f"Mock response to: {text}"
        
        def caption_image(self, image):
            return "Mock image caption"
        
        def transcribe_audio(self, audio):
            return {'text': 'Mock transcription', 'language': 'en', 'error': None}
    
    mock_text = MockProcessor()
    mock_image = MockProcessor()
    mock_audio = MockProcessor()
    
    fusion = FusionEngine(mock_text, mock_image, mock_audio)
    
    print("\n" + "="*60)
    print("Test 1: Input Analysis")
    print("="*60)
    
    # Test different input combinations
    test_cases = [
        {'text': 'Hello', 'image': None, 'audio': None},
        {'text': None, 'image': 'dummy', 'audio': None},
        {'text': 'Question?', 'image': 'dummy', 'audio': None},
        {'text': 'Hello', 'image': 'dummy', 'audio': 'audio.mp3'},
    ]
    
    for i, inputs in enumerate(test_cases, 1):
        analysis = fusion.analyze_inputs(**inputs)
        print(f"\nCase {i}: {inputs}")
        print(f"  Modality count: {analysis['modality_count']}")
        print(f"  Primary modality: {analysis['primary_modality']}")
        print(f"  Complexity: {analysis['input_complexity']}")
    
    print("\n" + "="*60)
    print("Test 2: Capabilities")
    print("="*60)
    
    caps = fusion.get_capabilities()
    print(f"Supported modalities: {', '.join(caps['supported_modalities'])}")
    print(f"Fusion modes: {', '.join(caps['fusion_modes'])}")
    print(f"Current mode: {caps['current_mode']}")
    
    print("\n" + "="*60)
    print("Test 3: Personality Fusion (Mock)")
    print("="*60)
    
    # Mock personality manager
    class MockPersonalityManager:
        def __init__(self):
            self.current_personality = 'assistant'
        
        def get_system_prompt(self):
            return "You are a helpful assistant."
        
        def get_current_personality(self):
            return {'name': 'assistant', 'temperature': 0.7}
        
        def get_temperature(self):
            return 0.7
        
        def get_max_length(self):
            return 150
    
    # Mock OpenAI integration
    class MockOpenAIIntegration:
        def is_enabled(self):
            return True
        
        def analyze_image(self, image, question):
            return f"OpenAI analysis of image with question: {question}"
        
        def generate_text(self, prompt, personality_config=None):
            return f"OpenAI response to: {prompt}"
    
    mock_personality = MockPersonalityManager()
    mock_openai = MockOpenAIIntegration()
    
    # Test personality fusion
    result = fusion.fuse_with_personality(
        text="Hello there",
        personality_manager=mock_personality,
        openai_integration=mock_openai
    )
    
    print(f"Personality fusion result: {result['text_response']}")
    print(f"Used OpenAI: {result['metadata']['used_openai']}")
    
    print("\n" + "="*60)
    print("✅ Fusion Engine tests complete!")
    print("="*60)