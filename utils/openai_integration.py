"""
OpenAI Integration - Premium GPT-4o features (optional)
Part 3: NeuraFusion Premium Features
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import base64
from io import BytesIO

# Load environment variables
load_dotenv()

class OpenAIIntegration:
    """
    Optional premium features using OpenAI's GPT-4o.
    
    Features:
    - Advanced text generation with GPT-4o
    - Vision understanding with GPT-4o
    - Enhanced multimodal reasoning
    
    Note: Requires OPENAI_API_KEY environment variable
    """
    
    def __init__(self):
        """Initialize OpenAI integration."""
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.is_available = bool(self.api_key and self.api_key.strip())
        
        if self.is_available:
            try:
                self.client = OpenAI(api_key=self.api_key)
                # Test connection
                self.client.models.list()
                print("✅ OpenAI GPT-4o integration active")
            except Exception as e:
                print(f"⚠️ OpenAI API key invalid: {e}")
                self.is_available = False
                self.client = None
        else:
            self.client = None
            print("ℹ️ OpenAI integration disabled (no API key)")
    
    def is_enabled(self):
        """Check if OpenAI integration is available."""
        return self.is_available
    
    def generate_text(self, prompt, personality_config=None, max_tokens=500):
        """
        Generate text using GPT-4o.
        
        Args:
            prompt: Input prompt
            personality_config: Optional personality settings
            max_tokens: Maximum response length
        
        Returns:
            Generated text or error message
        """
        if not self.is_available:
            return "OpenAI integration not available. Please add OPENAI_API_KEY to .env file."
        
        try:
            # Build messages
            messages = []
            
            # Add system message if personality provided
            if personality_config:
                system_prompt = personality_config.get('system_prompt', '')
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Get temperature from personality or use default
            temperature = personality_config.get('temperature', 0.7) if personality_config else 0.7
            
            # Call GPT-4o
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"OpenAI API Error: {str(e)}"
    
    def analyze_image(self, image, question=None):
        """
        Analyze image using GPT-4o Vision.
        
        Args:
            image: PIL Image or image path
            question: Optional question about the image
        
        Returns:
            Analysis text or error message
        """
        if not self.is_available:
            return "OpenAI integration not available."
        
        try:
            # Convert image to base64
            if isinstance(image, str):
                # If it's a file path
                with open(image, 'rb') as f:
                    image_data = f.read()
            else:
                # If it's a PIL Image
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                image_data = buffer.getvalue()
            
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Build prompt
            if question:
                prompt = question
            else:
                prompt = "Describe this image in detail. What do you see?"
            
            # Call GPT-4o Vision
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"OpenAI Vision API Error: {str(e)}"
    
    def multimodal_analysis(self, text=None, image=None, context=None):
        """
        Advanced multimodal analysis using GPT-4o.
        
        Args:
            text: Text input
            image: Image input (PIL Image or path)
            context: Additional context
        
        Returns:
            Analysis result
        """
        if not self.is_available:
            return "OpenAI integration not available."
        
        try:
            messages = []
            
            # Add context if provided
            if context:
                messages.append({
                    "role": "system",
                    "content": f"Context: {context}"
                })
            
            # Build multimodal message
            content = []
            
            if text:
                content.append({
                    "type": "text",
                    "text": text
                })
            
            if image:
                # Convert image to base64
                if isinstance(image, str):
                    with open(image, 'rb') as f:
                        image_data = f.read()
                else:
                    buffer = BytesIO()
                    image.save(buffer, format='PNG')
                    image_data = buffer.getvalue()
                
                base64_image = base64.b64encode(image_data).decode('utf-8')
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            
            if not content:
                return "No input provided"
            
            messages.append({
                "role": "user",
                "content": content
            })
            
            # Call GPT-4o
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=700
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"OpenAI Multimodal API Error: {str(e)}"
    
    def get_cost_estimate(self, input_tokens, output_tokens):
        """
        Estimate cost for GPT-4o usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            Cost estimate in USD
        """
        # GPT-4o pricing (as of 2024)
        input_cost_per_1k = 0.0025   # $2.50 per 1M tokens
        output_cost_per_1k = 0.01    # $10.00 per 1M tokens
        
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
    
    def get_status(self):
        """
        Get integration status information.
        
        Returns:
            Status dictionary
        """
        return {
            'enabled': self.is_available,
            'model': 'gpt-4o' if self.is_available else None,
            'features': [
                'Advanced text generation',
                'Vision understanding',
                'Multimodal reasoning'
            ] if self.is_available else [],
            'message': 'Active and ready' if self.is_available else 'Add OPENAI_API_KEY to enable'
        }


# Test function
if __name__ == "__main__":
    print("🧪 Testing OpenAI Integration...")
    print("="*70)
    
    integration = OpenAIIntegration()
    
    print("\n" + "="*70)
    print("Integration Status")
    print("="*70)
    
    status = integration.get_status()
    print(f"Enabled: {status['enabled']}")
    print(f"Model: {status['model']}")
    print(f"Message: {status['message']}")
    
    if status['enabled']:
        print("\nFeatures:")
        for feature in status['features']:
            print(f"  ✓ {feature}")
        
        print("\n" + "="*70)
        print("Test: Text Generation")
        print("="*70)
        
        response = integration.generate_text(
            "Explain quantum computing in simple terms.",
            max_tokens=150
        )
        print(f"Response: {response[:200]}...")
        
        print("\n" + "="*70)
        print("Cost Estimate Example")
        print("="*70)
        
        cost = integration.get_cost_estimate(input_tokens=100, output_tokens=150)
        print(f"Estimated cost for 100 input + 150 output tokens: ${cost:.6f}")
    else:
        print("\n💡 To enable OpenAI features:")
        print("  1. Get API key from: https://platform.openai.com/api-keys")
        print("  2. Add to .env file: OPENAI_API_KEY=sk-your-key-here")
        print("  3. Restart the application")
    
    print("\n" + "="*70)
    print("✅ OpenAI Integration test complete!")
    print("="*70)