"""
Visualization Module - Attention maps and image analysis visualizations
Part 3: NeuraFusion Advanced Visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import cv2

class AttentionVisualizer:
    """
    Creates visualizations for image analysis.
    
    Features:
    - Attention heatmaps (Grad-CAM style)
    - Color distribution analysis
    - Object detection overlays
    - Feature importance visualization
    """
    
    def __init__(self):
        """Initialize visualizer with default settings."""
        self.default_colormap = 'jet'
        self.default_alpha = 0.4
        
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        print("✅ Attention Visualizer initialized")
    
    def create_attention_heatmap(self, image, attention_weights=None, title="Attention Map"):
        """
        Create a heatmap overlay showing where the model is "looking".
        
        Args:
            image: PIL Image or numpy array
            attention_weights: 2D array of attention scores (optional)
                              If None, creates a simulated attention map
            title: Title for the visualization
        
        Returns:
            PIL Image with heatmap overlay
        """
        # Convert to numpy if PIL Image
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # If no attention weights provided, create simulated ones
        # In a real implementation, these would come from the model
        if attention_weights is None:
            attention_weights = self._simulate_attention(img_array.shape[:2])
        
        # Resize attention to match image
        if attention_weights.shape != img_array.shape[:2]:
            attention_weights = cv2.resize(
                attention_weights,
                (img_array.shape[1], img_array.shape[0])
            )
        
        # Normalize attention weights
        attention_norm = (attention_weights - attention_weights.min()) / \
                        (attention_weights.max() - attention_weights.min() + 1e-8)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Attention heatmap
        im = axes[1].imshow(attention_norm, cmap=self.default_colormap)
        axes[1].set_title("Attention Weights")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Overlay
        axes[2].imshow(img_array)
        axes[2].imshow(attention_norm, cmap=self.default_colormap, 
                      alpha=self.default_alpha)
        axes[2].set_title("Attention Overlay")
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close(fig)
        
        return result_image
    
    def _simulate_attention(self, shape):
        """
        Simulate attention weights for demonstration.
        In production, this would be replaced with actual model attention.
        
        Args:
            shape: Tuple (height, width)
        
        Returns:
            2D numpy array of attention weights
        """
        h, w = shape
        
        # Create center-focused attention (common pattern)
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        # Gaussian-like attention centered on image
        attention = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 3)**2))
        
        # Add some random variation
        noise = np.random.randn(h, w) * 0.1
        attention = attention + noise
        attention = np.clip(attention, 0, 1)
        
        return attention
    
    def analyze_color_distribution(self, image, title="Color Analysis"):
        """
        Analyze and visualize color distribution in image.
        
        Args:
            image: PIL Image or numpy array
            title: Title for visualization
        
        Returns:
            PIL Image with color analysis
        """
        # Convert to numpy
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(img_array)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # RGB histograms
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            axes[0, 1].hist(img_array[:, :, i].ravel(), bins=50, 
                          color=color, alpha=0.6, label=color.upper())
        axes[0, 1].set_title("RGB Histogram")
        axes[0, 1].set_xlabel("Pixel Value")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()
        
        # Color channels
        channel_names = ['Red', 'Green', 'Blue']
        for i in range(3):
            axes[0, 2].imshow(img_array[:, :, i], cmap='gray')
            axes[0, 2].set_title(f"{channel_names[i]} Channel")
            axes[0, 2].axis('off')
        
        # Dominant colors (simplified)
        reshaped = img_array.reshape(-1, 3)
        
        # Sample for speed
        if len(reshaped) > 10000:
            indices = np.random.choice(len(reshaped), 10000, replace=False)
            reshaped = reshaped[indices]
        
        # Show color scatter
        axes[1, 0].scatter(reshaped[:, 0], reshaped[:, 1], 
                          c=reshaped/255.0, alpha=0.1, s=1)
        axes[1, 0].set_title("Red vs Green")
        axes[1, 0].set_xlabel("Red")
        axes[1, 0].set_ylabel("Green")
        
        # Brightness histogram
        brightness = np.mean(img_array, axis=2)
        axes[1, 1].hist(brightness.ravel(), bins=50, color='gray', alpha=0.7)
        axes[1, 1].set_title("Brightness Distribution")
        axes[1, 1].set_xlabel("Brightness")
        axes[1, 1].set_ylabel("Frequency")
        
        # Color statistics
        stats_text = [
            f"Image Shape: {img_array.shape}",
            f"Mean RGB: ({img_array[:,:,0].mean():.1f}, "
            f"{img_array[:,:,1].mean():.1f}, {img_array[:,:,2].mean():.1f})",
            f"Std RGB: ({img_array[:,:,0].std():.1f}, "
            f"{img_array[:,:,1].std():.1f}, {img_array[:,:,2].std():.1f})",
            f"Brightness: {brightness.mean():.1f} ± {brightness.std():.1f}"
        ]
        
        axes[1, 2].axis('off')
        axes[1, 2].text(0.1, 0.5, '\n'.join(stats_text), 
                       fontsize=12, verticalalignment='center',
                       family='monospace')
        axes[1, 2].set_title("Statistics")
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close(fig)
        
        return result_image
    
    def create_comparison_grid(self, images, labels=None, title="Image Comparison"):
        """
        Create a grid comparing multiple images.
        
        Args:
            images: List of PIL Images or numpy arrays
            labels: List of labels for each image
            title: Overall title
        
        Returns:
            PIL Image with comparison grid
        """
        n_images = len(images)
        
        # Calculate grid size
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        # Flatten axes for easy iteration
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Plot each image
        for i, img in enumerate(images):
            if i < len(axes):
                if isinstance(img, Image.Image):
                    img_array = np.array(img)
                else:
                    img_array = img
                
                axes[i].imshow(img_array)
                if labels and i < len(labels):
                    axes[i].set_title(labels[i])
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close(fig)
        
        return result_image
    
    def visualize_image_features(self, image, features_dict):
        """
        Visualize extracted features from an image.
        
        Args:
            image: PIL Image
            features_dict: Dictionary of features to visualize
        
        Returns:
            PIL Image with feature visualization
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Original image
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax1.imshow(img_array)
        ax1.set_title("Original Image", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Features text
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        
        feature_text = "Extracted Features:\n\n"
        for key, value in features_dict.items():
            if isinstance(value, (int, float)):
                feature_text += f"{key}: {value:.2f}\n"
            elif isinstance(value, str):
                feature_text += f"{key}: {value}\n"
            else:
                feature_text += f"{key}: {str(value)[:30]}...\n"
        
        ax2.text(0.1, 0.9, feature_text, fontsize=11, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_title("Feature Summary", fontsize=14, fontweight='bold')
        
        # Feature importance (if available)
        ax3 = fig.add_subplot(gs[1, 1])
        if 'importance_scores' in features_dict:
            scores = features_dict['importance_scores']
            ax3.barh(range(len(scores)), list(scores.values()))
            ax3.set_yticks(range(len(scores)))
            ax3.set_yticklabels(list(scores.keys()))
            ax3.set_xlabel("Importance Score")
            ax3.set_title("Feature Importance", fontsize=12, fontweight='bold')
        else:
            ax3.axis('off')
            ax3.text(0.5, 0.5, "No importance scores available", 
                    ha='center', va='center')
        
        # Additional visualization space
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        ax4.text(0.5, 0.5, "NeuraFusion Advanced Image Analysis", 
                ha='center', va='center', fontsize=12, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.suptitle("Image Feature Visualization", fontsize=16, fontweight='bold')
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close(fig)
        
        return result_image


# Test function
if __name__ == "__main__":
    print("🧪 Testing Attention Visualizer...")
    print("="*70)
    
    visualizer = AttentionVisualizer()
    
    # Create test image
    from PIL import Image, ImageDraw
    
    print("\n" + "="*70)
    print("Test 1: Creating Test Image")
    print("="*70)
    
    test_img = Image.new('RGB', (400, 300), color='skyblue')
    draw = ImageDraw.Draw(test_img)
    
    # Draw shapes
    draw.rectangle([100, 100, 300, 200], fill='red', outline='black', width=3)
    draw.ellipse([50, 50, 150, 150], fill='yellow', outline='orange', width=2)
    draw.polygon([(350, 250), (380, 280), (320, 280)], fill='green')
    
    test_img.save("test_viz_image.jpg")
    print("✅ Test image created")
    
    print("\n" + "="*70)
    print("Test 2: Attention Heatmap")
    print("="*70)
    
    attention_result = visualizer.create_attention_heatmap(
        test_img, 
        title="Test Attention Visualization"
    )
    attention_result.save("test_attention.jpg")
    print("✅ Attention heatmap saved: test_attention.jpg")
    
    print("\n" + "="*70)
    print("Test 3: Color Distribution Analysis")
    print("="*70)
    
    color_result = visualizer.analyze_color_distribution(
        test_img,
        title="Test Color Analysis"
    )
    color_result.save("test_colors.jpg")
    print("✅ Color analysis saved: test_colors.jpg")
    
    print("\n" + "="*70)
    print("Test 4: Feature Visualization")
    print("="*70)
    
    test_features = {
        'dominant_color': 'Blue',
        'brightness': 128.5,
        'contrast': 0.75,
        'sharpness': 0.82,
        'complexity': 'Medium',
        'importance_scores': {
            'Color': 0.85,
            'Texture': 0.65,
            'Shape': 0.90,
            'Contrast': 0.70
        }
    }
    
    feature_result = visualizer.visualize_image_features(test_img, test_features)
    feature_result.save("test_features.jpg")
    print("✅ Feature visualization saved: test_features.jpg")
    
    print("\n" + "="*70)
    print("Test 5: Comparison Grid")
    print("="*70)
    
    # Create multiple test images
    test_images = []
    labels = []
    
    for i, color in enumerate(['red', 'green', 'blue', 'yellow']):
        img = Image.new('RGB', (200, 200), color=color)
        test_images.append(img)
        labels.append(f"Image {i+1}: {color}")
    
    grid_result = visualizer.create_comparison_grid(
        test_images,
        labels=labels,
        title="Test Image Grid"
    )
    grid_result.save("test_grid.jpg")
    print("✅ Comparison grid saved: test_grid.jpg")
    
    print("\n" + "="*70)
    print("✅ All Visualization tests complete!")
    print("="*70)
    print("\n📁 Generated files:")
    print("  - test_viz_image.jpg")
    print("  - test_attention.jpg")
    print("  - test_colors.jpg")
    print("  - test_features.jpg")
    print("  - test_grid.jpg")