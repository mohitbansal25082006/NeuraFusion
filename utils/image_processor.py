"""
Image Processing Module - Handles image understanding using BLIP-2
Fixed version with proper token length handling
"""

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

class ImageProcessor:
    """
    Manages image understanding using Salesforce's BLIP-2 model.
    
    BLIP-2 is a vision-language model that can:
    - Caption images (describe what's in them)
    - Answer questions about images
    - Perform visual reasoning
    """
    
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", cache_dir="./models_cache"):
        """
        Initialize the image processor.
        
        Args:
            model_name: Hugging Face model ID
                - 'blip2-opt-2.7b' = 2.7B parameters (good quality, medium speed)
                - 'blip2-flan-t5-xl' = Alternative with different backbone
            cache_dir: Where to save downloaded model weights
        """
        print(f"🔄 Loading image model: {model_name}")
        print("⏳ First-time download may take 5-7 minutes (larger model)...")
        
        # Load processor (handles image preprocessing)
        self.processor = Blip2Processor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Load the model
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32  # CPU compatibility
        )
        
        # Set to evaluation mode
        self.model.eval()
        
        print("✅ Image model loaded successfully!")
    
    def caption_image(self, image_path):
        """
        Generate a caption describing the image.
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            Generated caption string
        """
        
        # Load image if path is provided
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")
        
        # Process image for the model
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        
        # Generate caption with proper token handling
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,      # Generate up to 50 new tokens
                num_beams=5,            # Beam search for better quality
                min_length=10,          # Minimum 10 tokens (prevents too short captions)
                length_penalty=1.0,     # Neutral length preference
                repetition_penalty=1.5  # Discourage repetition
            )
        
        # Decode to text
        caption = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        return caption.strip()
    
    def answer_question(self, image_path, question):
        """
        Answer a question about an image (Visual Question Answering).
        
        Args:
            image_path: Path to image file or PIL Image object
            question: Question to ask about the image
            
        Returns:
            Answer string
        """
        
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")
        
        # Create prompt with question
        prompt = f"Question: {question} Answer:"
        
        # Process both image and text
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )
        
        # Generate answer with proper token handling
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,     # Allow longer answers for complex questions
                num_beams=5,            # Beam search for quality
                min_length=5,           # At least 5 tokens
                length_penalty=1.0,     # Neutral length preference
                repetition_penalty=1.5, # Discourage repetition
                no_repeat_ngram_size=3  # Don't repeat 3-word phrases
            )
        
        # Decode answer
        answer = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        return answer.strip()
    
    def analyze_image(self, image_path, query=None):
        """
        Main interface for Gradio - handles both captioning and Q&A.
        
        Args:
            image_path: Path to image or PIL Image
            query: Optional question. If None, generates caption
            
        Returns:
            Analysis result string
        """
        if query and len(query.strip()) > 0:
            # User asked a specific question
            return self.answer_question(image_path, query)
        else:
            # Just caption the image
            return self.caption_image(image_path)
    
    def batch_analyze(self, images, queries=None):
        """
        Analyze multiple images at once (more efficient than one-by-one).
        
        Args:
            images: List of image paths or PIL Images
            queries: Optional list of questions (same length as images, or None)
            
        Returns:
            List of analysis results
        """
        results = []
        
        # Convert all to PIL Images
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img.convert("RGB"))
        
        # If no queries, batch caption all images
        if queries is None or all(q is None or len(q.strip()) == 0 for q in queries):
            inputs = self.processor(
                images=pil_images,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=5,
                    min_length=10
                )
            
            # Decode all captions
            for i in range(len(generated_ids)):
                caption = self.processor.decode(
                    generated_ids[i],
                    skip_special_tokens=True
                )
                results.append(caption.strip())
        else:
            # Process each image-query pair individually
            for img, query in zip(pil_images, queries):
                if query and len(query.strip()) > 0:
                    result = self.answer_question(img, query)
                else:
                    result = self.caption_image(img)
                results.append(result)
        
        return results
    
    def get_detailed_description(self, image_path):
        """
        Get a detailed, comprehensive description of an image.
        Uses multiple prompts to extract rich information.
        
        Args:
            image_path: Path to image or PIL Image
            
        Returns:
            Detailed description string
        """
        
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")
        
        # Get basic caption
        caption = self.caption_image(image)
        
        # Ask specific questions for more details
        questions = [
            "What is the main subject or object in this image?",
            "What colors are prominent in this image?",
            "What is the setting or environment?",
            "Are there any people or animals? If so, what are they doing?"
        ]
        
        details = [f"**Overview:** {caption}\n"]
        
        for question in questions:
            try:
                answer = self.answer_question(image, question)
                details.append(f"**{question}** {answer}")
            except:
                continue
        
        return "\n\n".join(details)


# Test function
if __name__ == "__main__":
    print("🧪 Testing Image Processor...")
    print("⚠️  This test requires a sample image.")
    print("Creating a simple test image...")
    
    # Create a simple test image
    from PIL import Image, ImageDraw, ImageFont
    
    # Create colorful test image
    test_img = Image.new('RGB', (400, 300), color='skyblue')
    draw = ImageDraw.Draw(test_img)
    
    # Draw some shapes
    draw.rectangle([100, 100, 300, 200], fill='red', outline='black', width=3)
    draw.ellipse([50, 50, 150, 150], fill='yellow', outline='orange', width=2)
    draw.polygon([(350, 250), (380, 280), (320, 280)], fill='green')
    
    # Add text
    try:
        draw.text((150, 140), "TEST", fill='white')
    except:
        pass  # If font not available, skip text
    
    test_img.save("test_image.jpg")
    print("✅ Test image created: test_image.jpg")
    
    print("\n" + "="*60)
    processor = ImageProcessor()
    print("="*60)
    
    # Test 1: Captioning
    print("\n📸 Test 1: Image Captioning")
    print("-" * 60)
    caption = processor.caption_image("test_image.jpg")
    print(f"Caption: {caption}")
    
    # Test 2: Visual Q&A
    print("\n❓ Test 2: Visual Question Answering")
    print("-" * 60)
    
    test_questions = [
        "What color is the rectangle?",
        "What shapes are in the image?",
        "Is there a circle in this image?"
    ]
    
    for q in test_questions:
        print(f"\nQ: {q}")
        answer = processor.answer_question("test_image.jpg", q)
        print(f"A: {answer}")
    
    # Test 3: Detailed Description
    print("\n📋 Test 3: Detailed Description")
    print("-" * 60)
    detailed = processor.get_detailed_description("test_image.jpg")
    print(detailed)
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
    print("\n💡 If you see reasonable descriptions above, the model is working!")
    print("🎉 You can now run the main app: python app.py")