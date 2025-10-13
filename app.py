"""
NeuraFusion - Part 1: Text & Image Processing Interface
A multimodal AI assistant powered by Hugging Face models
"""

import gradio as gr
import json
from pathlib import Path

# Import our custom utilities
from utils.text_processor import TextProcessor
from utils.image_processor import ImageProcessor

# Load configuration
with open("config/model_configs.json", "r") as f:
    config = json.load(f)

# Initialize processors (loaded once when app starts)
print("🚀 Initializing NeuraFusion...")
print("=" * 50)

text_processor = TextProcessor(
    model_name=config["text_model"]["name"],
    cache_dir=config["cache_directory"]
)

image_processor = ImageProcessor(
    model_name=config["image_model"]["name"],
    cache_dir=config["cache_directory"]
)

print("=" * 50)
print("✅ All models loaded! Starting UI...")

# ============================================
# GRADIO INTERFACE FUNCTIONS
# ============================================

def text_chat_interface(message, history):
    """
    Handle text-only chat conversations.
    
    Args:
        message: Current user message
        history: List of previous [user, bot] message pairs
        
    Returns:
        Updated history with new message pair
    """
    if not message or len(message.strip()) == 0:
        return history
    
    # Get response from the model
    response = text_processor.chat(message, history)
    
    # Append to history as [user_message, bot_response] pair
    history = history + [[message, response]]
    
    return history


def image_analysis_interface(image, question):
    """
    Handle image understanding (captioning or Q&A).
    
    Args:
        image: PIL Image object from Gradio
        question: Optional question about the image
        
    Returns:
        Analysis result string
    """
    if image is None:
        return "Please upload an image!"
    
    result = image_processor.analyze_image(image, question)
    return result


def multimodal_interface(text_input, image_input):
    """
    Handle combined text + image inputs.
    
    Args:
        text_input: User's text query
        image_input: Uploaded image
        
    Returns:
        Combined response string
    """
    if image_input is None and not text_input:
        return "Please provide either text, an image, or both!"
    
    response_parts = []
    
    # Process image if provided
    if image_input is not None:
        image_caption = image_processor.caption_image(image_input)
        response_parts.append(f"📸 **Image Analysis:**\n{image_caption}")
        
        # If text query relates to image, answer about the image
        if text_input and len(text_input.strip()) > 0:
            image_answer = image_processor.answer_question(image_input, text_input)
            response_parts.append(f"\n\n💬 **Answer about the image:**\n{image_answer}")
    
    # Process text query if provided (and no image, or as additional context)
    if text_input and len(text_input.strip()) > 0 and image_input is None:
        text_response = text_processor.generate_response(text_input)
        response_parts.append(f"💬 **Text Response:**\n{text_response}")
    
    return "\n".join(response_parts)


# ============================================
# BUILD GRADIO INTERFACE
# ============================================

# Custom CSS for dark theme with purple accents
custom_css = """
#header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
}

#header h1 {
    color: white;
    font-size: 2.5em;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

#header p {
    color: rgba(255,255,255,0.9);
    font-size: 1.1em;
    margin-top: 10px;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}

footer {
    text-align: center;
    margin-top: 30px;
    padding: 20px;
    color: #888;
}
"""

# Create the main interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # Header
    with gr.Row(elem_id="header"):
        gr.HTML("""
            <h1>🧠 NeuraFusion</h1>
            <p>Advanced Multimodal AI Assistant - Part 1: Text & Image Processing</p>
        """)
    
    # Main tabs
    with gr.Tabs():
        
        # ========== TAB 1: TEXT CHAT ==========
        with gr.Tab("💬 Text Chat"):
            gr.Markdown("""
            ### Chat with AI
            Ask questions, get explanations, or have a conversation!
            
            **Examples to try:**
            - "Explain how neural networks work"
            - "Write a creative story about a robot"
            - "What is the capital of France and why is it famous?"
            """)
            
            chatbot = gr.Chatbot(
                height=500,
                label="Conversation",
                show_label=True
            )
            
            with gr.Row():
                txt_input = gr.Textbox(
                    placeholder="Type your message here...",
                    label="Your message",
                    scale=4
                )
                txt_submit = gr.Button("Send 📤", scale=1, variant="primary")
            
            txt_clear = gr.Button("Clear Conversation 🗑️")
            
            # Event handlers
            txt_submit.click(
                text_chat_interface,
                inputs=[txt_input, chatbot],
                outputs=[chatbot]
            ).then(
                lambda: "",  # Clear input after sending
                outputs=[txt_input]
            )
            
            txt_input.submit(  # Also trigger on Enter key
                text_chat_interface,
                inputs=[txt_input, chatbot],
                outputs=[chatbot]
            ).then(
                lambda: "",
                outputs=[txt_input]
            )
            
            txt_clear.click(lambda: None, outputs=[chatbot])
        
        
        # ========== TAB 2: IMAGE ANALYSIS ==========
        with gr.Tab("🖼️ Image Understanding"):
            gr.Markdown("""
            ### Analyze Images
            Upload an image and optionally ask a question about it!
            
            **What you can do:**
            - Get automatic image captions
            - Ask questions: "What colors are in this image?"
            - Request details: "How many people are in the photo?"
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        type="pil",
                        label="Upload Image",
                        height=400
                    )
                    img_question = gr.Textbox(
                        placeholder="Ask a question about the image (optional)...",
                        label="Question (optional)",
                        lines=2
                    )
                    img_submit = gr.Button("Analyze 🔍", variant="primary")
                
                with gr.Column(scale=1):
                    img_output = gr.Textbox(
                        label="Analysis Result",
                        lines=15,
                        show_copy_button=True
                    )
            
            # Event handler
            img_submit.click(
                image_analysis_interface,
                inputs=[img_input, img_question],
                outputs=[img_output]
            )
        
        
        # ========== TAB 3: MULTIMODAL ==========
        with gr.Tab("🔗 Multimodal Fusion"):
            gr.Markdown("""
            ### Combine Text + Image
            Use both text and images together for richer interactions!
            
            **Examples:**
            - Upload image + ask: "What's the main object in this image?"
            - Upload image without text → Get automatic description
            - Text only → Regular chat response
            """)
            
            with gr.Row():
                with gr.Column():
                    mm_text = gr.Textbox(
                        placeholder="Enter your text query here...",
                        label="Text Input",
                        lines=3
                    )
                    mm_image = gr.Image(
                        type="pil",
                        label="Upload Image (optional)",
                        height=300
                    )
                    mm_submit = gr.Button("Process 🚀", variant="primary", size="lg")
                
                with gr.Column():
                    mm_output = gr.Textbox(
                        label="Response",
                        lines=20,
                        show_copy_button=True
                    )
            
            # Event handler
            mm_submit.click(
                multimodal_interface,
                inputs=[mm_text, mm_image],
                outputs=[mm_output]
            )
        
        
        # ========== TAB 4: ABOUT ==========
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            # About NeuraFusion - Part 1
            
            ## 🎯 What is this?
            NeuraFusion is an advanced multimodal AI assistant that understands text and images.
            This is **Part 1** of a 3-part project series.
            
            ## 🛠️ Technology Stack
            
            | Component | Technology | Purpose |
            |-----------|-----------|---------|
            | **Text Understanding** | Google Flan-T5 Base (250M params) | Conversational AI |
            | **Image Understanding** | Salesforce BLIP-2 (2.7B params) | Image captioning & VQA |
            | **UI Framework** | Gradio 5.x | Interactive web interface |
            | **Backend** | Hugging Face Transformers | Model loading & inference |
            
            ## ✨ Features in Part 1
            
            ✅ **Text Chat:** Have natural conversations with AI  
            ✅ **Image Captioning:** Automatic image descriptions  
            ✅ **Visual Q&A:** Ask questions about images  
            ✅ **Multimodal Fusion:** Combine text and images  
            
            ## 📊 Model Information
            
            ### Flan-T5 Base
            - **Parameters:** 250 million
            - **Training:** Instruction-tuned on 1,800+ tasks
            - **Strengths:** General knowledge, reasoning, summarization
            
            ### BLIP-2 OPT-2.7B
            - **Parameters:** 2.7 billion
            - **Training:** 129M image-text pairs
            - **Strengths:** Image captioning, visual question answering
            
            ## 🔜 Coming in Part 2
            
            - 🎤 Voice input (Whisper speech-to-text)
            - 🔊 Voice output (Text-to-speech)
            - 🧠 Enhanced multimodal fusion
            - 💾 Conversation memory
            
            ## 📝 Credits
            
            Built with ❤️ using:
            - Hugging Face Transformers
            - Gradio
            - PyTorch
            
            ---
            
            **Cost:** $0 (100% Free Tier)
            
            Made with 🧠 by [Your Name]
            """)
    
    # Footer
    with gr.Row():
        gr.Markdown("""
        <footer>
        <p>🧠 <strong>NeuraFusion v1.0</strong> | Powered by Hugging Face 🤗</p>
        <p style="font-size: 0.9em; color: #666;">
        All models run locally on your machine. No data is sent to external servers.
        </p>
        </footer>
        """)

# ============================================
# LAUNCH THE APP
# ============================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎉 NeuraFusion is ready!")
    print("="*50)
    print("\n💡 The web interface will open in your browser automatically.")
    print("📍 Default URL: http://127.0.0.1:7860")
    print("\n⌨️  Press Ctrl+C in the terminal to stop the server.\n")
    
    demo.launch(
        server_name="127.0.0.1",  # Only accessible from your computer
        server_port=7860,         # Port number
        share=False,              # Don't create public link (for local testing)
        show_error=True,          # Show detailed errors for debugging
        inbrowser=True            # Automatically open in browser
    )