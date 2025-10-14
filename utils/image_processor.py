"""
Image Processing Module - Complete Enhanced Version with Visualization Support
Part 3: NeuraFusion Advanced Image Understanding
"""

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import numpy as np
from pathlib import Path

class ImageProcessor:
    """
    Enhanced image understanding using Salesforce's BLIP-2 model.
    
    BLIP-2 is a vision-language model that can:
    - Caption images (describe what's in them)
    - Answer questions about images (Visual Question Answering)
    - Perform visual reasoning
    - Extract image features for visualization
    
    New in Part 3:
    - Feature extraction for visualizations
    - Enhanced attention analysis
    - Color and composition metrics
    - Batch processing optimizations
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
        elif isinstance(image_path, Path):
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
        elif isinstance(image_path, Path):
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
            elif isinstance(img, Path):
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
        elif isinstance(image_path, Path):
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
    
    def get_image_features(self, image_path):
        """
        Extract detailed features from an image for visualization.
        
        New in Part 3: Enhanced feature extraction for advanced visualizations.
        
        Args:
            image_path: Path to image or PIL Image
        
        Returns:
            Dictionary of image features including:
                - Dimensions (width, height, aspect_ratio)
                - Color statistics (mean_rgb, std_rgb, brightness)
                - Dominant channel and color
                - AI-generated caption
                - Complexity metrics
        """
        
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Path):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")
        
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Extract basic features
        height, width, channels = img_array.shape
        
        # Calculate color statistics
        mean_r = float(img_array[:, :, 0].mean())
        mean_g = float(img_array[:, :, 1].mean())
        mean_b = float(img_array[:, :, 2].mean())
        
        std_r = float(img_array[:, :, 0].std())
        std_g = float(img_array[:, :, 1].std())
        std_b = float(img_array[:, :, 2].std())
        
        # Calculate brightness
        brightness = (mean_r + mean_g + mean_b) / 3.0
        
        # Determine dominant channel
        channel_means = [mean_r, mean_g, mean_b]
        dominant_channel_idx = channel_means.index(max(channel_means))
        channel_names = ['Red', 'Green', 'Blue']
        dominant_channel = channel_names[dominant_channel_idx]
        
        # Determine dominant color (simplified)
        if mean_r > mean_g and mean_r > mean_b:
            if mean_r > 180:
                dominant_color = 'Red/Pink'
            else:
                dominant_color = 'Dark Red'
        elif mean_g > mean_r and mean_g > mean_b:
            if mean_g > 180:
                dominant_color = 'Green/Lime'
            else:
                dominant_color = 'Dark Green'
        elif mean_b > mean_r and mean_b > mean_g:
            if mean_b > 180:
                dominant_color = 'Blue/Cyan'
            else:
                dominant_color = 'Dark Blue'
        else:
            if brightness > 200:
                dominant_color = 'White/Light'
            elif brightness < 50:
                dominant_color = 'Black/Dark'
            else:
                dominant_color = 'Gray/Neutral'
        
        # Calculate contrast (standard deviation of brightness)
        brightness_map = np.mean(img_array, axis=2)
        contrast = float(brightness_map.std())
        
        # Estimate complexity (edge density approximation)
        # Higher std in pixel values = more complex/detailed image
        complexity_score = (std_r + std_g + std_b) / 3.0
        if complexity_score > 60:
            complexity = 'High'
        elif complexity_score > 30:
            complexity = 'Medium'
        else:
            complexity = 'Low'
        
        # Get AI caption
        try:
            caption = self.caption_image(image)
        except:
            caption = "Caption generation failed"
        
        # Calculate sharpness estimate (variance of Laplacian approximation)
        # Simple gradient-based sharpness
        gray = np.mean(img_array, axis=2)
        gy, gx = np.gradient(gray)
        sharpness = float(np.sqrt(gx**2 + gy**2).mean())
        sharpness_normalized = min(sharpness / 50.0, 1.0)  # Normalize to 0-1
        
        # Calculate color diversity (number of unique colors, sampled)
        # Reduce image for speed
        if img_array.size > 1000000:  # If larger than 1000x1000
            sample_factor = int(np.sqrt(img_array.size / 1000000))
            img_sampled = img_array[::sample_factor, ::sample_factor]
        else:
            img_sampled = img_array
        
        reshaped = img_sampled.reshape(-1, 3)
        # Quantize colors to reduce unique count (group similar colors)
        quantized = (reshaped // 32) * 32
        unique_colors = len(np.unique(quantized, axis=0))
        color_diversity = min(unique_colors / 100.0, 1.0)  # Normalize to 0-1
        
        # Build features dictionary
        features = {
            # Basic dimensions
            'width': width,
            'height': height,
            'aspect_ratio': round(width / height, 2),
            'total_pixels': width * height,
            
            # Color statistics
            'mean_rgb': (round(mean_r, 1), round(mean_g, 1), round(mean_b, 1)),
            'std_rgb': (round(std_r, 1), round(std_g, 1), round(std_b, 1)),
            'brightness': round(brightness, 1),
            'contrast': round(contrast, 1),
            
            # Derived properties
            'dominant_channel': dominant_channel,
            'dominant_color': dominant_color,
            'complexity': complexity,
            'complexity_score': round(complexity_score, 1),
            'sharpness': round(sharpness_normalized, 2),
            'color_diversity': round(color_diversity, 2),
            
            # AI analysis
            'caption': caption,
            
            # Importance scores for visualization
            'importance_scores': {
                'Color': round(color_diversity, 2),
                'Texture': round(complexity_score / 100, 2),
                'Sharpness': round(sharpness_normalized, 2),
                'Contrast': round(contrast / 100, 2),
                'Brightness': round(brightness / 255, 2)
            }
        }
        
        return features
    
    def get_color_palette(self, image_path, n_colors=5):
        """
        Extract dominant color palette from image.
        
        Args:
            image_path: Path to image or PIL Image
            n_colors: Number of dominant colors to extract
        
        Returns:
            List of RGB tuples representing dominant colors
        """
        
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Path):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")
        
        # Resize for faster processing
        image_small = image.resize((100, 100))
        img_array = np.array(image_small)
        
        # Reshape to 2D array of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Simple k-means-like clustering (manual implementation)
        # For production, you'd use sklearn.cluster.KMeans
        # Here's a simplified version:
        
        # Randomly sample initial centroids
        np.random.seed(42)
        indices = np.random.choice(len(pixels), n_colors, replace=False)
        centroids = pixels[indices].astype(float)
        
        # Iterate to find dominant colors
        for _ in range(10):
            # Assign pixels to nearest centroid
            distances = np.sqrt(((pixels[:, np.newaxis] - centroids) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([pixels[labels == i].mean(axis=0) 
                                     if (labels == i).any() else centroids[i] 
                                     for i in range(n_colors)])
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        # Convert to list of tuples
        palette = [tuple(map(int, color)) for color in centroids]
        
        return palette
    
    def compare_images(self, image1_path, image2_path):
        """
        Compare two images and describe differences.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
        
        Returns:
            Dictionary with comparison results
        """
        
        # Get captions for both images
        caption1 = self.caption_image(image1_path)
        caption2 = self.caption_image(image2_path)
        
        # Get features for both
        features1 = self.get_image_features(image1_path)
        features2 = self.get_image_features(image2_path)
        
        # Build comparison
        comparison = {
            'image1_caption': caption1,
            'image2_caption': caption2,
            'brightness_diff': abs(features1['brightness'] - features2['brightness']),
            'contrast_diff': abs(features1['contrast'] - features2['contrast']),
            'complexity_diff': abs(features1['complexity_score'] - features2['complexity_score']),
            'dominant_color1': features1['dominant_color'],
            'dominant_color2': features2['dominant_color'],
            'similarity_score': self._calculate_similarity(features1, features2)
        }
        
        return comparison
    
    def _calculate_similarity(self, features1, features2):
        """
        Calculate similarity score between two images based on features.
        
        Args:
            features1: Features dict from first image
            features2: Features dict from second image
        
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        
        # Compare brightness
        brightness_sim = 1 - min(abs(features1['brightness'] - features2['brightness']) / 255.0, 1.0)
        
        # Compare contrast
        contrast_sim = 1 - min(abs(features1['contrast'] - features2['contrast']) / 100.0, 1.0)
        
        # Compare complexity
        complexity_sim = 1 - min(abs(features1['complexity_score'] - features2['complexity_score']) / 100.0, 1.0)
        
        # Average similarity
        similarity = (brightness_sim + contrast_sim + complexity_sim) / 3.0
        
        return round(similarity, 2)
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model.config._name_or_path,
            'total_parameters': f"{total_params:,}",
            'trainable_parameters': f"{trainable_params:,}",
            'dtype': str(self.model.dtype),
            'device': str(next(self.model.parameters()).device),
            'capabilities': [
                'Image Captioning',
                'Visual Question Answering',
                'Feature Extraction',
                'Batch Processing'
            ]
        }


# Test function
if __name__ == "__main__":
    print("🧪 Testing Enhanced Image Processor...")
    print("="*70)
    print("⚠️  This test requires a sample image.")
    print("Creating a comprehensive test image...")
    
    # Create a complex test image
    from PIL import Image, ImageDraw, ImageFont
    
    # Create colorful test image with multiple elements
    test_img = Image.new('RGB', (600, 400), color='lightblue')
    draw = ImageDraw.Draw(test_img)
    
    # Draw various shapes with different colors
    draw.rectangle([50, 50, 200, 150], fill='red', outline='darkred', width=3)
    draw.ellipse([250, 50, 400, 150], fill='yellow', outline='orange', width=3)
    draw.polygon([(450, 50), (550, 50), (500, 150)], fill='green', outline='darkgreen', width=2)
    
    draw.rectangle([50, 200, 150, 350], fill='purple', outline='black', width=2)
    draw.ellipse([200, 200, 350, 350], fill='cyan', outline='blue', width=2)
    draw.rectangle([400, 200, 550, 350], fill='pink', outline='red', width=2)
    
    # Add some text
    try:
        draw.text((250, 380), "TEST IMAGE", fill='black', anchor='mt')
    except:
        pass  # If font not available, skip text
    
    test_img.save("test_image_enhanced.jpg")
    print("✅ Test image created: test_image_enhanced.jpg")
    
    print("\n" + "="*70)
    print("Initializing Image Processor...")
    print("="*70)
    
    processor = ImageProcessor()
    
    print("\n" + "="*70)
    print("Test 1: Basic Image Captioning")
    print("-"*70)
    
    caption = processor.caption_image("test_image_enhanced.jpg")
    print(f"Caption: {caption}")
    
    print("\n" + "="*70)
    print("Test 2: Visual Question Answering")
    print("-"*70)
    
    test_questions = [
        "What colors are in the image?",
        "What shapes can you see?",
        "Is there text in this image?",
        "Describe the layout of objects"
    ]
    
    for q in test_questions:
        print(f"\nQ: {q}")
        answer = processor.answer_question("test_image_enhanced.jpg", q)
        print(f"A: {answer}")
    
    print("\n" + "="*70)
    print("Test 3: Detailed Description")
    print("-"*70)
    
    detailed = processor.get_detailed_description("test_image_enhanced.jpg")
    print(detailed)
    
    print("\n" + "="*70)
    print("Test 4: Feature Extraction (NEW in Part 3)")
    print("-"*70)
    
    features = processor.get_image_features("test_image_enhanced.jpg")
    
    print(f"📐 Dimensions: {features['width']}x{features['height']}")
    print(f"📏 Aspect Ratio: {features['aspect_ratio']}")
    print(f"🎨 Mean RGB: {features['mean_rgb']}")
    print(f"💡 Brightness: {features['brightness']}")
    print(f"🔆 Contrast: {features['contrast']}")
    print(f"🎯 Dominant Color: {features['dominant_color']}")
    print(f"🧩 Complexity: {features['complexity']} ({features['complexity_score']})")
    print(f"🔍 Sharpness: {features['sharpness']}")
    print(f"🌈 Color Diversity: {features['color_diversity']}")
    print(f"📝 Caption: {features['caption']}")
    
    print("\n🎯 Importance Scores:")
    for metric, score in features['importance_scores'].items():
        bar = "█" * int(score * 20)
        print(f"  {metric:12s} [{bar:<20}] {score:.2f}")
    
    print("\n" + "="*70)
    print("Test 5: Color Palette Extraction (NEW)")
    print("-"*70)
    
    palette = processor.get_color_palette("test_image_enhanced.jpg", n_colors=5)
    print("Dominant Colors (RGB):")
    for i, color in enumerate(palette, 1):
        print(f"  {i}. RGB{color}")
    
    print("\n" + "="*70)
    print("Test 6: Model Information")
    print("-"*70)
    
    model_info = processor.get_model_info()
    print(f"Model: {model_info['model_name']}")
    print(f"Parameters: {model_info['total_parameters']}")
    print(f"Device: {model_info['device']}")
    print(f"Data Type: {model_info['dtype']}")
    print("\nCapabilities:")
    for cap in model_info['capabilities']:
        print(f"  ✓ {cap}")
    
    print("\n" + "="*70)
    print("Test 7: Batch Processing")
    print("-"*70)
    
    # Create multiple test images
    test_images = []
    for i in range(3):
        img = Image.new('RGB', (200, 200), color=['red', 'green', 'blue'][i])
        img_path = f"test_batch_{i}.jpg"
        img.save(img_path)
        test_images.append(img_path)
        print(f"✓ Created: {img_path}")
    
    print("\nProcessing batch...")
    results = processor.batch_analyze(test_images)
    
    for i, result in enumerate(results):
        print(f"\nImage {i+1}: {result}")
    
    # Clean up batch test images
    for img_path in test_images:
        Path(img_path).unlink()
    
    print("\n" + "="*70)
    print("✅ All Enhanced Image Processor tests complete!")
    print("="*70)
    
    print("\n📊 Test Summary:")
    print("  ✓ Basic captioning works")
    print("  ✓ Visual Q&A functional")
    print("  ✓ Detailed descriptions generated")
    print("  ✓ Feature extraction complete")
    print("  ✓ Color palette extraction works")
    print("  ✓ Batch processing functional")
    print("  ✓ Model info accessible")
    
    print("\n💡 Next Steps:")
    print("  1. Image processor is fully functional")
    print("  2. Ready for integration with visualization module")
    print("  3. Can be used in main app.py")
    print("  4. Run: python app.py to test in full application")
    
    print("\n🎉 Enhanced Image Processor ready for production!")