"""
Text Processing Module - Enhanced with personality support
Part 3: NeuraFusion Advanced Text Processing
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class TextProcessor:
    """
    Enhanced text processor with personality mode support.
    
    New in Part 3:
    - Personality-aware response generation
    - Context enhancement
    - Improved prompt engineering
    """
    
    def __init__(self, model_name="google/flan-t5-base", cache_dir="./models_cache"):
        """Initialize text processor."""
        print(f"🔄 Loading text model: {model_name}")
        print("⏳ First-time download may take 2-3 minutes...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32
        )
        
        self.model.eval()
        
        print("✅ Text model loaded successfully!")
    
    def generate_response(self, user_input, max_length=512, temperature=0.7, 
                         personality_context=None):
        """
        Generate response with optional personality context.
        
        Args:
            user_input: User's question or prompt
            max_length: Maximum response length
            temperature: Randomness (0.0-1.0)
            personality_context: Optional personality system prompt
        
        Returns:
            Generated text response
        """
        
        # Build enhanced prompt
        if personality_context:
            prompt = f"{personality_context}\n\nQuestion: {user_input}\n\nAnswer:"
        else:
            prompt = f"Answer the following question concisely and helpfully:\n\nQuestion: {user_input}\n\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def generate_with_context(self, user_input, conversation_history=None, 
                            personality_context=None, max_length=512, temperature=0.7):
        """
        Generate response with conversation history context.
        
        Args:
            user_input: Current user message
            conversation_history: List of previous messages
            personality_context: Personality system prompt
            max_length: Max response length
            temperature: Randomness
        
        Returns:
            Generated response
        """
        
        # Build context-aware prompt
        prompt_parts = []
        
        if personality_context:
            prompt_parts.append(personality_context)
        
        if conversation_history:
            prompt_parts.append("Previous conversation:")
            for msg in conversation_history[-3:]:  # Last 3 turns
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                prompt_parts.append(f"{role.capitalize()}: {content}")
        
        prompt_parts.append(f"\nCurrent question: {user_input}\n\nAnswer:")
        
        full_prompt = "\n".join(prompt_parts)
        
        return self.generate_response(
            full_prompt,
            max_length=max_length,
            temperature=temperature
        )
    
    def chat(self, message, history=None, personality_context=None):
        """
        Gradio-compatible chat function.
        
        Args:
            message: Current user message
            history: Gradio chat history
            personality_context: Optional personality prompt
        
        Returns:
            Bot response
        """
        response = self.generate_response(
            message,
            personality_context=personality_context
        )
        return response


# Test remains the same
if __name__ == "__main__":
    print("🧪 Testing Enhanced Text Processor...")
    processor = TextProcessor()
    
    test_questions = [
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms."
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        answer = processor.generate_response(question)
        print(f"💬 Answer: {answer}")