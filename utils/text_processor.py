"""
Text Processing Module - Handles text generation using Flan-T5
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class TextProcessor:
    """
    Manages text-based AI conversation using Google's Flan-T5 model.
    
    Flan-T5 is a free, instruction-tuned model that can:
    - Answer questions
    - Summarize text
    - Translate languages
    - Perform reasoning tasks
    """
    
    def __init__(self, model_name="google/flan-t5-base", cache_dir="./models_cache"):
        """
        Initialize the text processor.
        
        Args:
            model_name: Hugging Face model ID (default: flan-t5-base)
                - 'base' = 250M parameters (good balance of speed/quality)
                - 'small' = 80M parameters (faster but less capable)
                - 'large' = 780M parameters (slower but more capable)
            cache_dir: Where to save downloaded model weights
        """
        print(f"🔄 Loading text model: {model_name}")
        print("⏳ First-time download may take 2-3 minutes...")
        
        # Load tokenizer (converts text to numbers the model understands)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Load the actual model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32  # Use float32 for CPU compatibility
        )
        
        # Set model to evaluation mode (not training)
        self.model.eval()
        
        print("✅ Text model loaded successfully!")
    
    def generate_response(self, user_input, max_length=512, temperature=0.7):
        """
        Generate a response to user input.
        
        Args:
            user_input: The user's question or prompt
            max_length: Maximum response length in tokens
            temperature: Randomness (0.0=deterministic, 1.0=creative)
            
        Returns:
            Generated text response
        """
        
        # Prepare the prompt with instruction format
        # Flan-T5 works best with clear instructions
        prompt = f"Answer the following question concisely and helpfully:\n\nQuestion: {user_input}\n\nAnswer:"
        
        # Tokenize input (convert text to model-readable format)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",  # Return PyTorch tensors
            max_length=512,
            truncation=True  # Cut off if too long
        )
        
        # Generate response using the model
        with torch.no_grad():  # Don't calculate gradients (saves memory)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,  # Use sampling for varied responses
                top_p=0.9,  # Nucleus sampling (quality control)
                num_return_sequences=1
            )
        
        # Decode the generated tokens back to text
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True  # Remove <pad>, </s> tokens
        )
        
        return response.strip()
    
    def chat(self, message, history=None):
        """
        Gradio-compatible chat function with conversation history.
        
        Args:
            message: Current user message
            history: List of [user_msg, bot_msg] pairs (Gradio format)
            
        Returns:
            Bot response string
        """
        # For Part 1, we'll just respond to current message
        # In Part 2, we'll use the history for context
        response = self.generate_response(message)
        return response


# Test function (run this file directly to test)
if __name__ == "__main__":
    print("🧪 Testing Text Processor...")
    processor = TextProcessor()
    
    test_questions = [
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about technology."
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        answer = processor.generate_response(question)
        print(f"💬 Answer: {answer}")