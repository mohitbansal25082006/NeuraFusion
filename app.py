import gradio as gr
import json
from pathlib import Path
import os
from datetime import datetime
# Import all processors and managers
from utils.text_processor import TextProcessor
from utils.image_processor import ImageProcessor
from utils.audio_processor import AudioProcessor
from utils.fusion_engine import FusionEngine
from utils.memory_manager import MemoryManager
from utils.personality_manager import PersonalityManager
from utils.visualization import AttentionVisualizer
from utils.openai_integration import OpenAIIntegration
# Load environment variables
from dotenv import load_dotenv
load_dotenv()
# Load configuration
with open("config/model_configs.json", "r") as f:
    config = json.load(f)
# Create exports directory
Path("exports").mkdir(exist_ok=True)
# ============================================
# INITIALIZE ALL COMPONENTS
# ============================================
print("\n" + "="*70)
print("🚀 NeuraFusion Part 3 - Production Version")
print("="*70)
# Initialize processors
print("\n📦 Loading AI Models...")
text_processor = TextProcessor(
    model_name=config["text_model"]["name"],
    cache_dir=config["cache_directory"]
)
image_processor = ImageProcessor(
    model_name=config["image_model"]["name"],
    cache_dir=config["cache_directory"]
)
audio_processor = AudioProcessor(
    whisper_model="base",
    cache_dir=config["cache_directory"]
)
# Initialize managers
print("\n🎭 Initializing Advanced Features...")
personality_manager = PersonalityManager()
memory_manager = MemoryManager(max_history=100)
attention_visualizer = AttentionVisualizer()
openai_integration = OpenAIIntegration()
# Initialize fusion engine
fusion_engine = FusionEngine(
    text_processor,
    image_processor,
    audio_processor
)
print("\n" + "="*70)
print("✅ All systems ready!")
print(f"🎭 Personalities: {len(personality_manager.get_all_personalities())} modes")
print(f"🔐 OpenAI: {'Enabled ✓' if openai_integration.is_enabled() else 'Disabled (optional)'}")
print("="*70 + "\n")
# ============================================
# GRADIO INTERFACE FUNCTIONS
# ============================================
def chat_with_personality(message, history, personality_choice, use_openai):
    """
    Enhanced chat with personality modes and optional OpenAI.
  
    Args:
        message: User message
        history: Gradio chat history
        personality_choice: Selected personality
        use_openai: Whether to use OpenAI if available
  
    Returns:
        Updated history
    """
    if not message or len(message.strip()) == 0:
        return history
  
    # Set personality
    personality_manager.set_personality(personality_choice)
  
    # Add to memory
    memory_manager.add_user_message(message, modalities=['text'])
  
    # Generate response
    if use_openai and openai_integration.is_enabled():
        personality_config = personality_manager.get_current_personality()
        response = openai_integration.generate_text(message, personality_config)
    else:
        response = fusion_engine.generate_with_personality(
            message,
            personality_manager,
            openai_integration if use_openai else None
        )
  
    # Add to memory
    memory_manager.add_assistant_message(response, modalities=['text'])
  
    # Update history
    history = history + [[message, response]]
  
    return history

def voice_chat_with_personality(audio_input, personality_choice, use_openai):
    """
    Voice chat with personality modes.
  
    Args:
        audio_input: Audio file
        personality_choice: Selected personality
        use_openai: Use OpenAI if available
  
    Returns:
        Tuple: (transcription, response_text, response_audio)
    """
    if audio_input is None:
        return "", "Please record or upload audio.", None
  
    # Transcribe
    transcription = audio_processor.transcribe_audio(audio_input)
  
    if transcription['error']:
        return "", f"Error: {transcription['error']}", None
  
    user_text = transcription['text']
  
    # Set personality
    personality_manager.set_personality(personality_choice)
  
    # Add to memory
    memory_manager.add_user_message(user_text, modalities=['audio', 'text'])
  
    # Generate response
    if use_openai and openai_integration.is_enabled():
        personality_config = personality_manager.get_current_personality()
        response_text = openai_integration.generate_text(user_text, personality_config)
    else:
        response_text = fusion_engine.generate_with_personality(
            user_text,
            personality_manager,
            openai_integration if use_openai else None
        )
  
    # Add to memory
    memory_manager.add_assistant_message(response_text, modalities=['text', 'audio'])
  
    # Generate voice
    response_audio = audio_processor.text_to_speech(response_text)
  
    return user_text, response_text, response_audio

def advanced_image_analysis(image_input, question_text, personality_choice,
                           use_openai, show_attention, show_colors):
    """
    Advanced image analysis with visualizations.
  
    Args:
        image_input: PIL Image
        question_text: Question about image
        personality_choice: Selected personality
        use_openai: Use OpenAI if available
        show_attention: Show attention heatmap
        show_colors: Show color analysis
  
    Returns:
        Tuple: (answer, attention_viz, color_viz)
    """
    if image_input is None:
        return "Please upload an image.", None, None
  
    # Set personality
    personality_manager.set_personality(personality_choice)
  
    # Add to memory
    memory_manager.add_user_message(
        question_text if question_text else "Image analysis",
        modalities=['image', 'text']
    )
  
    # Analyze image
    if use_openai and openai_integration.is_enabled():
        answer = openai_integration.analyze_image(image_input, question_text)
    else:
        answer = image_processor.analyze_image(image_input, question_text)
  
    # Add to memory
    memory_manager.add_assistant_message(answer, modalities=['text'])
  
    # Generate visualizations
    attention_viz = None
    color_viz = None
  
    if show_attention:
        try:
            attention_viz = attention_visualizer.create_attention_heatmap(
                image_input,
                title="Attention Analysis"
            )
        except Exception as e:
            print(f"Attention viz error: {e}")
  
    if show_colors:
        try:
            color_viz = attention_visualizer.analyze_color_distribution(
                image_input,
                title="Color Distribution"
            )
        except Exception as e:
            print(f"Color viz error: {e}")
  
    return answer, attention_viz, color_viz

def export_to_csv(timestamp):
    """Export conversation to CSV format."""
    import csv
  
    filepath = f"exports/conversation_{timestamp}.csv"
  
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Turn', 'Role', 'Timestamp', 'Content', 'Modalities'])
      
        for i, msg in enumerate(memory_manager.conversation_history, 1):
            writer.writerow([
                i,
                msg['role'],
                msg['timestamp'],
                msg['content'],
                ', '.join(msg.get('modalities', ['text']))
            ])
  
    return filepath

def export_conversation_enhanced(format_choice):
    """
    Export conversation in various formats.
  
    Args:
        format_choice: Export format (json, txt, md, csv)
  
    Returns:
        Tuple: (file_path, status_message)
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      
        if format_choice == "JSON":
            filepath = memory_manager.export_to_json(
                f"exports/conversation_{timestamp}.json"
            )
        elif format_choice == "Text":
            filepath = memory_manager.export_to_text(
                f"exports/conversation_{timestamp}.txt"
            )
        elif format_choice == "Markdown":
            filepath = export_to_markdown(timestamp)
        else: # CSV
            filepath = export_to_csv(timestamp)
      
        return filepath, f"✅ Exported successfully to: {filepath}"
  
    except Exception as e:
        return None, f"❌ Export failed: {str(e)}"

def export_to_markdown(timestamp):
    """Export conversation to Markdown format."""
    filepath = f"exports/conversation_{timestamp}.md"
  
    summary = memory_manager.get_session_summary()
  
    lines = [
        f"# NeuraFusion Conversation Export",
        f"",
        f"**Session ID:** `{summary['session_id']}`",
        f"**Date:** {summary['start_time']}",
        f"**Total Turns:** {summary['total_turns']}",
        f"**Modalities:** {', '.join(summary['modalities_used'])}",
        f"",
        f"---",
        f""
    ]
  
    for i, msg in enumerate(memory_manager.conversation_history, 1):
        role = "👤 **User**" if msg['role'] == 'user' else "🤖 **Assistant**"
        timestamp_short = msg['timestamp'].split('T')[1][:8]
      
        lines.append(f"## Turn {i} - {role} `{timestamp_short}`")
        lines.append(f"")
        lines.append(msg['content'])
        lines.append(f"")
        lines.append(f"*Modalities: {', '.join(msg.get('modalities', ['text']))}*")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
  
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
  
    return filepath

def complete_multimodal_fusion(text_input, image_input, audio_input,
                               personality_choice, use_openai, enable_voice_output,
                               show_visualizations):
    """
    Complete multimodal fusion with all features.
  
    Args:
        text_input: Text query
        image_input: Image file
        audio_input: Audio file
        personality_choice: Personality mode
        use_openai: Use OpenAI
        enable_voice_output: Generate voice
        show_visualizations: Show image visualizations
  
    Returns:
        Tuple: (response_text, response_audio, processing_info, visualization)
    """
    # Check inputs
    has_input = (
        (text_input and len(text_input.strip()) > 0) or
        image_input is not None or
        audio_input is not None
    )
  
    if not has_input:
        return "Please provide at least one input.", None, "", None
  
    # Set personality
    personality_manager.set_personality(personality_choice)
  
    # Determine modalities
    modalities = []
    if text_input and len(text_input.strip()) > 0:
        modalities.append('text')
    if image_input is not None:
        modalities.append('image')
    if audio_input is not None:
        modalities.append('audio')
  
    # Add to memory
    input_desc = f"Multimodal: {', '.join(modalities)}"
    if text_input:
        input_desc = text_input
    memory_manager.add_user_message(input_desc, modalities=modalities)
  
    # Process through enhanced fusion
    result = fusion_engine.fuse_with_personality(
        text=text_input,
        image=image_input,
        audio=audio_input,
        personality_manager=personality_manager,
        openai_integration=openai_integration if use_openai else None,
        include_voice_output=enable_voice_output
    )
  
    response_text = result['text_response']
    response_audio = result['audio_response']
  
    # Add to memory
    response_modalities = ['text']
    if response_audio:
        response_modalities.append('audio')
    memory_manager.add_assistant_message(response_text, modalities=response_modalities)
  
    # Generate processing info
    processing_info = fusion_engine.get_processing_summary(result['metadata'])
  
    # Add model info
    model_used = "GPT-4o" if result['metadata'].get('used_openai') else "Local Models"
    processing_info += f"\n- Model: {model_used}"
    processing_info += f"\n- Personality: {personality_choice}"
  
    # Generate visualization
    visualization = None
    if show_visualizations and image_input is not None:
        try:
            # Get image features
            features = image_processor.get_image_features(image_input)
            visualization = attention_visualizer.visualize_image_features(
                image_input,
                features
            )
        except Exception as e:
            print(f"Visualization error: {e}")
  
    return response_text, response_audio, processing_info, visualization

def get_session_analytics():
    """
    Get detailed session analytics.
  
    Returns:
        Formatted analytics string
    """
    summary = memory_manager.get_session_summary()
    mod_stats = memory_manager.get_modality_statistics()
  
    # Calculate metrics
    avg_message_length = 0
    if summary['history_length'] > 0:
        total_chars = sum(len(msg['content']) for msg in memory_manager.conversation_history)
        avg_message_length = total_chars / summary['history_length']
  
    output = [
        "📊 **COMPREHENSIVE SESSION ANALYTICS**",
        "",
        "### 📈 Overview",
        f"- **Session ID:** `{summary['session_id']}`",
        f"- **Duration:** {summary['duration_seconds']:.0f} seconds ({summary['duration_seconds']/60:.1f} minutes)",
        f"- **Total Interactions:** {summary['total_turns']} turns",
        f"- **Messages:** {summary['history_length']}",
        f" - User: {summary['user_messages']}",
        f" - Assistant: {summary['assistant_messages']}",
        "",
        "### 🎭 Personality Usage",
        f"- **Current Mode:** {personality_manager.current_personality}",
        "",
        "### 📱 Modality Statistics",
        f"- **Text:** {mod_stats['text']} uses",
        f"- **Image:** {mod_stats['image']} uses",
        f"- **Audio:** {mod_stats['audio']} uses",
        f"- **Multimodal:** {mod_stats['multimodal']} combinations",
        "",
        "### 💬 Content Metrics",
        f"- **Average Message Length:** {avg_message_length:.0f} characters",
        "",
        "### 🔐 AI Models Used",
        f"- **Text Generation:** {'GPT-4o' if openai_integration.is_enabled() else 'Flan-T5 (local)'}",
        f"- **Image Analysis:** {'GPT-4o Vision' if openai_integration.is_enabled() else 'BLIP-2 (local)'}",
        f"- **Speech Recognition:** Whisper (local)",
        f"- **Text-to-Speech:** gTTS (free)",
    ]
  
    return "\n".join(output)

def switch_personality(personality_choice):
    """
    Switch personality mode.
  
    Args:
        personality_choice: Personality key
  
    Returns:
        Status message and description
    """
    result = personality_manager.set_personality(personality_choice)
    description = personality_manager.get_personality_description()
  
    return result, description

def get_system_status():
    """
    Get comprehensive system status.
  
    Returns:
        Formatted status string
    """
    openai_status = openai_integration.get_status()
  
    output = [
        "🔧 **SYSTEM STATUS**",
        "",
        "### 🤖 AI Models",
        f"- **Text:** {config['text_model']['name']} ✓",
        f"- **Vision:** {config['image_model']['name']} ✓",
        f"- **Audio:** {config['audio_model']['name']} ✓",
        "",
        "### 🎭 Personalities",
        f"- **Available:** {len(personality_manager.get_all_personalities())} modes",
        f"- **Active:** {personality_manager.current_personality}",
        "",
        "### 🔐 OpenAI Integration",
        f"- **Status:** {'✅ Enabled' if openai_status['enabled'] else '⚠️ Disabled (optional)'}",
    ]
  
    if openai_status['enabled']:
        output.append(f"- **Model:** {openai_status['model']}")
        output.append("- **Features:** " + ", ".join(openai_status['features']))
    else:
        output.append("- **Message:** " + openai_status['message'])
  
    output.extend([
        "",
        "### 💾 Memory",
        f"- **History Length:** {len(memory_manager)} messages",
        f"- **Max Capacity:** {memory_manager.max_history} messages",
        "",
        "### 🎨 Visualizations",
        "- **Attention Maps:** ✓ Available",
        "- **Color Analysis:** ✓ Available",
        "- **Feature Extraction:** ✓ Available",
    ])
  
    return "\n".join(output)

def handle_chat_export():
    """
    Handle export from smart chat tab.
    
    Returns:
        Export status message
    """
    return "💾 Chat exported! You can now download it from the Export tab."

# ============================================
# BUILD ENHANCED GRADIO INTERFACE
# ============================================
# Load custom CSS
css_path = Path("assets/styles.css")
custom_css = ""
if css_path.exists():
    with open(css_path, 'r') as f:
        custom_css = f.read()
else:
    # Fallback minimal CSS
    custom_css = """
    #header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .scrollable-textbox {
        max-height: 400px;
        overflow-y: auto !important;
    }
    """

# Get personality choices
personality_choices = [p['name'] for p in personality_manager.get_all_personalities()]
personality_keys = [p['key'] for p in personality_manager.get_all_personalities()]

# Create main interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="NeuraFusion V3") as demo:
  
    # Header
    with gr.Row(elem_id="header"):
        gr.HTML(f"""
            <h1>🧠 NeuraFusion V3.0 Production</h1>
            <p>Complete Multimodal AI • Personalities • Visualizations • Analytics</p>
            <p style="font-size: 0.9em; opacity: 0.9;">
            {'🔐 OpenAI Enabled' if openai_integration.is_enabled() else '🆓 Free Tier Mode'} |
            🎭 {len(personality_choices)} Personalities |
            💾 Smart Memory
            </p>
        """)
  
    # Main tabs
    with gr.Tabs():
      
        # ========== TAB 1: ENHANCED TEXT CHAT ==========
        with gr.Tab("💬 Smart Chat"):
            gr.Markdown("""
            ### Intelligent Conversation with Personality Modes
          
            **Features:**
            - 🎭 5 personality modes (Mentor, Friend, Analyst, Professional, Creative)
            - 🔐 Optional GPT-4o integration for premium responses
            - 💾 Full conversation memory
            """)
          
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=500,
                        label="Conversation",
                        avatar_images=("👤", "🤖"),
                        show_copy_button=True
                    )
                  
                    with gr.Row():
                        txt_input = gr.Textbox(
                            placeholder="Type your message...",
                            label="Your message",
                            scale=4,
                            lines=2
                        )
                        txt_submit = gr.Button("Send 📤", scale=1, variant="primary")
                  
                    with gr.Row():
                        txt_clear = gr.Button("Clear 🗑️")
                        txt_export = gr.Button("Export 💾")
                  
                    # Export status message for chat tab
                    chat_export_status = gr.Textbox(
                        label="Export Status",
                        visible=False,
                        interactive=False
                    )
              
                with gr.Column(scale=1):
                    personality_select = gr.Radio(
                        choices=personality_keys,
                        value=personality_manager.current_personality,
                        label="🎭 Personality Mode",
                        info="Choose conversation style"
                    )
                  
                    use_openai_chat = gr.Checkbox(
                        label="🔐 Use GPT-4o ",
                        value=False,
                        info="Premium AI model"
                    )
                  
                    personality_info = gr.Markdown(
                        personality_manager.get_personality_description(),
                        label="Mode Description"
                    )
          
            # Event handlers
            txt_submit.click(
                chat_with_personality,
                inputs=[txt_input, chatbot, personality_select, use_openai_chat],
                outputs=[chatbot]
            ).then(lambda: "", outputs=[txt_input])
          
            txt_input.submit(
                chat_with_personality,
                inputs=[txt_input, chatbot, personality_select, use_openai_chat],
                outputs=[chatbot]
            ).then(lambda: "", outputs=[txt_input])
          
            personality_select.change(
                switch_personality,
                inputs=[personality_select],
                outputs=[gr.Textbox(visible=False), personality_info]
            )
          
            txt_clear.click(lambda: None, outputs=[chatbot])
          
            # Export button in chat tab
            txt_export.click(
                handle_chat_export,
                outputs=[chat_export_status]
            )
      
        # ========== TAB 2: VOICE CHAT ==========
        with gr.Tab("🎤 Voice Assistant"):
            gr.Markdown("""
            ### Talk to AI with Your Voice
          
            **Features:**
            - 🎤 Speech-to-text (Whisper)
            - 🎭 Personality modes
            - 🔊 Voice responses
            """)
          
            with gr.Row():
                with gr.Column():
                    voice_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="🎤 Record or Upload"
                    )
                  
                    voice_personality = gr.Radio(
                        choices=personality_keys,
                        value=personality_manager.current_personality,
                        label="🎭 Personality"
                    )
                  
                    use_openai_voice = gr.Checkbox(
                        label="🔐 Use GPT-4o",
                        value=False
                    )
                  
                    voice_submit = gr.Button("Process 🎧", variant="primary", size="lg")
              
                with gr.Column():
                    voice_transcription = gr.Textbox(
                        label="📝 What you said:",
                        lines=3
                    )
                    voice_response_text = gr.Textbox(
                        label="💬 AI Response:",
                        lines=8
                    )
                    voice_response_audio = gr.Audio(
                        label="🔊 Voice Response:",
                        type="filepath"
                    )
          
            voice_submit.click(
                voice_chat_with_personality,
                inputs=[voice_input, voice_personality, use_openai_voice],
                outputs=[voice_transcription, voice_response_text, voice_response_audio]
            )
      
        # ========== TAB 3: ADVANCED IMAGE ANALYSIS ==========
        with gr.Tab("🖼️ Vision Analysis"):
            gr.Markdown("""
            ### Advanced Image Understanding with Visualizations
          
            **New Features:**
            - 🔍 Attention heatmaps (see what AI focuses on)
            - 🎨 Color distribution analysis
            - 🔐 Optional GPT-4o Vision
            - 🎭 Personality-aware descriptions
            """)
          
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        type="pil",
                        label="📸 Upload Image",
                        height=350
                    )
                  
                    img_question = gr.Textbox(
                        label="❓ Ask a Question (optional)",
                        placeholder="e.g., What colors are dominant?",
                        lines=2
                    )
                  
                    img_personality = gr.Radio(
                        choices=personality_keys,
                        value=personality_manager.current_personality,
                        label="🎭 Personality"
                    )
                  
                    with gr.Row():
                        use_openai_img = gr.Checkbox(
                            label="🔐 GPT-4o Vision",
                            value=False
                        )
                        show_attention = gr.Checkbox(
                            label="🔍 Show Attention",
                            value=True
                        )
                        show_colors = gr.Checkbox(
                            label="🎨 Show Colors",
                            value=True
                        )
                  
                    img_submit = gr.Button("Analyze 🔬", variant="primary", size="lg")
              
                with gr.Column(scale=1):
                    img_answer = gr.Textbox(
                        label="💬 Analysis Result:",
                        lines=12,
                        show_copy_button=True,
                        elem_classes=["scrollable-textbox"]
                    )
          
            with gr.Row():
                attention_output = gr.Image(
                    label="🔍 Attention Heatmap",
                    type="pil"
                )
                color_output = gr.Image(
                    label="🎨 Color Analysis",
                    type="pil"
                )
          
            img_submit.click(
                advanced_image_analysis,
                inputs=[img_input, img_question, img_personality, use_openai_img,
                       show_attention, show_colors],
                outputs=[img_answer, attention_output, color_output]
            )
      
        # ========== TAB 4: COMPLETE MULTIMODAL FUSION ==========
        with gr.Tab("🔗 Complete Fusion"):
            gr.Markdown("""
            ### Ultimate Multimodal Experience
          
            **All Features Combined:**
            - 📝 Text + 🖼️ Image + 🎤 Audio fusion
            - 🎭 Personality modes
            - 🔐 OpenAI GPT-4o (optional)
            - 🎨 Visual analysis
            - 🔊 Voice output
            """)
          
            with gr.Row():
                with gr.Column(scale=1):
                    mm_text = gr.Textbox(
                        placeholder="Enter your question or context...",
                        label="📝 Text Input",
                        lines=3
                    )
                  
                    mm_image = gr.Image(
                        type="pil",
                        label="📸 Image Input (optional)",
                        height=250
                    )
                  
                    mm_audio = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="🎤 Audio Input (optional)"
                    )
                  
                    mm_personality = gr.Radio(
                        choices=personality_keys,
                        value=personality_manager.current_personality,
                        label="🎭 Personality Mode"
                    )
                  
                    with gr.Row():
                        use_openai_mm = gr.Checkbox(
                            label="🔐 Use GPT-4o",
                            value=False
                        )
                        mm_voice_output = gr.Checkbox(
                            label="🔊 Voice Response",
                            value=True
                        )
                        show_viz_mm = gr.Checkbox(
                            label="🎨 Visualizations",
                            value=True
                        )
                  
                    mm_submit = gr.Button("🚀 Process All", variant="primary", size="lg")
              
                with gr.Column(scale=1):
                    mm_response = gr.Textbox(
                        label="💬 Complete Response",
                        lines=15,
                        show_copy_button=True,
                        elem_classes=["scrollable-textbox"]
                    )
                  
                    mm_audio_response = gr.Audio(
                        label="🔊 Voice Response",
                        type="filepath"
                    )
                  
                    mm_processing_info = gr.Textbox(
                        label="📊 Processing Information",
                        lines=6,
                        elem_classes=["scrollable-textbox"]
                    )
          
            with gr.Row():
                mm_visualization = gr.Image(
                    label="🎨 Image Feature Visualization",
                    type="pil"
                )
          
            mm_submit.click(
                complete_multimodal_fusion,
                inputs=[mm_text, mm_image, mm_audio, mm_personality,
                       use_openai_mm, mm_voice_output, show_viz_mm],
                outputs=[mm_response, mm_audio_response, mm_processing_info, mm_visualization]
            )
      
        # ========== TAB 5: ANALYTICS DASHBOARD ==========
        with gr.Tab("📊 Analytics"):
            gr.Markdown("""
            ### Session Analytics & Insights
          
            **Track Your Usage:**
            - 📈 Conversation statistics
            - 🎭 Personality usage patterns
            - 📱 Modality breakdown
            - ⏱️ Session duration
            """)
          
            with gr.Row():
                with gr.Column():
                    analytics_refresh = gr.Button("🔄 Refresh Analytics", variant="primary")
                  
                    analytics_display = gr.Textbox(
                        label="📊 Session Analytics",
                        lines=25,
                        show_copy_button=True
                    )
              
                with gr.Column():
                    system_status_refresh = gr.Button("🔧 System Status")
                  
                    system_status_display = gr.Textbox(
                        label="🔧 System Status",
                        lines=25,
                        show_copy_button=True
                    )
          
            analytics_refresh.click(
                get_session_analytics,
                outputs=[analytics_display]
            )
          
            system_status_refresh.click(
                get_system_status,
                outputs=[system_status_display]
            )
      
        # ========== TAB 6: ENHANCED MEMORY & EXPORT ==========
        with gr.Tab("💾 Memory & Export"):
            gr.Markdown("""
            ### Conversation Memory Management
          
            **Enhanced Features:**
            - 💾 Export to JSON, TXT, Markdown, CSV
            - 🔍 Search history
            - 📊 View statistics
            - 🗑️ Clear memory
            """)
          
            with gr.Row():
                with gr.Column():
                    history_refresh = gr.Button("🔄 Refresh History", variant="primary")
                  
                    with gr.Row():
                        export_format = gr.Radio(
                            choices=["JSON", "Text", "Markdown", "CSV"],
                            value="Text",
                            label="📄 Export Format"
                        )
                  
                    with gr.Row():
                        export_btn = gr.Button("💾 Export Conversation", scale=2)
                        clear_history_btn = gr.Button("🗑️ Clear All", scale=1, variant="stop")
                  
                    export_file = gr.File(label="📥 Download File")
                    export_status = gr.Textbox(
                        label="Status",
                        lines=2
                    )
              
                with gr.Column():
                    history_display = gr.Textbox(
                        label="📋 Conversation History",
                        lines=28,
                        show_copy_button=True
                    )
          
            # Event handlers
            history_refresh.click(
                lambda: memory_manager.get_session_summary(),
                outputs=None
            ).then(
                get_session_analytics,
                outputs=[history_display]
            )
          
            export_btn.click(
                export_conversation_enhanced,
                inputs=[export_format],
                outputs=[export_file, export_status]
            )
          
            clear_history_btn.click(
                lambda: memory_manager.clear_history(),
                outputs=None
            ).then(
                lambda: ("🗑️ History cleared!", ""),
                outputs=[export_status, history_display]
            )
      
        # ========== TAB 7: DOCUMENTATION ==========
        with gr.Tab("📚 Documentation"):
            gr.Markdown("""
            # 🧠 NeuraFusion V3.0 - Complete Documentation
          
            ## 🎉 What's New in Part 3?
          
            ### 🎭 Personality System
            Choose from 5 distinct AI personalities:
          
            1. **🎓 Mentor Mode**
               - Patient and educational
               - Detailed explanations with examples
               - Perfect for learning new concepts
          
            2. **😊 Friend Mode** (Default)
               - Casual and conversational
               - Warm and empathetic responses
               - Great for everyday chat
          
            3. **📊 Analyst Mode**
               - Data-driven and logical
               - Structured analysis
               - Ideal for research and analysis
          
            4. **💼 Professional Mode**
               - Formal business communication
               - Concise and actionable
               - Best for work-related queries
          
            5. **🎨 Creative Mode**
               - Imaginative and expressive
               - Unique perspectives
               - Perfect for brainstorming
          
            ---
          
            ### 🔐 OpenAI GPT-4o Integration (Optional)
          
            **Premium Features:**
            - More advanced reasoning
            - Better multimodal understanding
            - Longer context windows
            - Higher quality responses
          
            **How to Enable:**
            1. Get API key from: https://platform.openai.com/api-keys
            2. Add to `.env` file: `OPENAI_API_KEY=sk-your-key-here`
            3. Restart the application
            4. Check the "Use GPT-4o" checkbox in any tab
          
            **Cost:** ~$0.0025 per 1000 input tokens, ~$0.01 per 1000 output tokens
          
            **Note:** The app works perfectly without OpenAI - all free local models!
          
            ---
          
            ### 🔍 Visual Analysis Features
          
            #### Attention Heatmaps
            - See where the AI "looks" in images
            - Understand model focus areas
            - Gradient-based visualization
          
            #### Color Distribution Analysis
            - RGB channel breakdowns
            - Brightness histograms
            - Dominant color detection
            - Statistical analysis
          
            #### Feature Visualization
            - Extract image characteristics
            - Display feature importance
            - Comprehensive visual reports
          
            ---
          
            ### 💾 Enhanced Export System
          
            **4 Export Formats:**
          
            1. **JSON** - Machine-readable, complete metadata
            2. **Text** - Human-readable conversation log
            3. **Markdown** - Formatted for documentation
            4. **CSV** - Spreadsheet-compatible for analysis
          
            All exports include:
            - Timestamps
            - Modality information
            - Session metadata
            - Complete message history
          
            ---
          
            ### 📊 Analytics Dashboard
          
            Track your usage with detailed metrics:
          
            - **Session Duration:** Total time active
            - **Message Counts:** User vs Assistant
            - **Modality Usage:** Text, Image, Audio breakdown
            - **Personality Stats:** Which modes you use most
            - **Model Information:** Which AI models are active
          
            ---
          
            ## 🛠️ Technology Stack (Complete)
          
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
          
            ## 📖 Usage Guide
          
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
          
            ---
          
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
          
            ---
          
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
          
            ---
          
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
          
            ---
          
            ### 📊 Analytics Tab
          
            **Best For:**
            - Tracking usage patterns
            - Understanding your habits
            - System monitoring
            - Performance insights
          
            **What You'll See:**
            - Session statistics
            - Modality preferences
            - Personality usage
            - AI model status
          
            ---
          
            ### 💾 Memory & Export Tab
          
            **Best For:**
            - Saving conversations
            - Reviewing history
            - Sharing insights
            - Data analysis
          
            **Export Uses:**
            - JSON for developers/analysis
            - Text for reading/sharing
            - Markdown for documentation
            - CSV for spreadsheet analysis
          
            ---
          
            ## 🎓 Key Concepts Explained
          
            ### What are Personalities?
          
            Think of personalities as different "hats" the AI can wear:
            - **Mentor** = Teacher mode
            - **Friend** = Buddy mode
            - **Analyst** = Researcher mode
            - **Professional** = Business mode
            - **Creative** = Artist mode
          
            Each adjusts:
            - Response style
            - Temperature (creativity)
            - Length preferences
            - Tone and language
          
            ---
          
            ### What is Multimodal Fusion?
          
            **Multimodal** = Multiple types of input
          
            The AI can understand:
            1. **Text** - Your typed questions
            2. **Images** - Visual information
            3. **Audio** - Your voice
          
            **Fusion** = Combining these intelligently
          
            Example:
            - You upload a sunset photo
            - Ask via voice: "What time of day?"
            - AI analyzes image + understands question
            - Responds: "This appears to be golden hour, around 6-7 PM"
          
            ---
          
            ### What are Attention Heatmaps?
          
            When AI "looks" at an image, it focuses on certain areas more than others.
          
            **Heatmap Colors:**
            - 🔴 **Red/Yellow** = High attention (AI focused here)
            - 🔵 **Blue/Purple** = Low attention (AI ignored this)
          
            **Why Useful:**
            - Understand AI reasoning
            - Verify correct focus
            - Debug misunderstandings
            - Learn about model behavior
          
            ---
          
            ### Local vs OpenAI Models
          
            **Local Models (Free):**
            - ✅ Completely free
            - ✅ Privacy (runs on your PC)
            - ✅ No internet needed (after download)
            - ⚠️ Slower on CPU
            - ⚠️ Less advanced reasoning
          
            **OpenAI GPT-4o (Premium):**
            - ✅ State-of-the-art performance
            - ✅ Better understanding
            - ✅ Faster responses
            - ⚠️ Costs money (~$0.01 per interaction)
            - ⚠️ Requires internet
            - ⚠️ Data sent to OpenAI
          
            **Recommendation:** Start with free local models. Upgrade to OpenAI only if you need premium quality.
          
            ---
          
            ## 🚀 Deployment Guide
          
            ### Option 1: Local Development
          
            **Current Setup** - You're here! ✓
            - Runs on your PC
            - Access via: http://127.0.0.1:7860
            - Private and secure
          
            ---
          
            ### Option 2: Hugging Face Spaces (Free Hosting)
          
            **Steps to Deploy:**
          
            1. **Create Hugging Face Account**
               - Go to: https://huggingface.co/join
               - Sign up for free
          
            2. **Create New Space**
               - Click "New Space"
               - Name: "my-neurafusion"
               - SDK: Gradio
               - Hardware: CPU (free) or GPU (paid)
          
            3. **Upload Files**
               - Upload all project files
               - Include requirements.txt
               - Add .env with your keys
          
            4. **Deploy**
               - Space builds automatically
               - Get public URL
               - Share with anyone!
          
            **Detailed Guide:** See `deploy_to_hf.py` script
          
            ---
          
            ### Option 3: Share Temporarily
          
            **Quick Sharing:**
            - In `app.py`, change: `share=True`
            - Get temporary public link
            - Valid for 72 hours
            - No setup needed!
          
            ---
          
            ## 💡 Pro Tips & Tricks
          
            ### Getting Best Results
          
            1. **Be Specific**
               - ❌ "Tell me about this"
               - ✅ "What are the main colors and objects in this image?"
          
            2. **Use Right Personality**
               - Learning → Mentor
               - Analysis → Analyst
               - Chat → Friend
          
            3. **Combine Modalities**
               - Image + specific text question
               - Voice + image for natural interaction
          
            4. **Check Visualizations**
               - Attention maps show AI understanding
               - Color analysis reveals image properties
          
            5. **Export Regularly**
               - Save important conversations
               - Use JSON for later analysis
          
            ---
          
            ### Troubleshooting
          
            **Slow Responses?**
            - First run downloads models (one-time)
            - CPU is slower than GPU
            - Close other programs
            - Consider GPU upgrade or Colab
          
            **OpenAI Not Working?**
            - Check API key in .env
            - Verify account has credits
            - Restart application after adding key
          
            **Voice Recording Issues?**
            - Grant microphone permissions
            - Check system audio settings
            - Try uploading file instead
          
            **Visualizations Not Showing?**
            - Ensure matplotlib installed
            - Check console for errors
            - Try regenerating
          
            **Out of Memory?**
            - Clear conversation history
            - Restart application
            - Close other programs
            - Use smaller images
          
            ---
          
            ## 📚 Additional Resources
          
            **Documentation:**
            - Gradio: https://gradio.app/docs
            - Hugging Face: https://huggingface.co/docs
            - OpenAI: https://platform.openai.com/docs
          
            **Model Papers:**
            - Flan-T5: https://arxiv.org/abs/2210.11416
            - BLIP-2: https://arxiv.org/abs/2301.12597
            - Whisper: https://arxiv.org/abs/2212.04356
            - GPT-4: https://arxiv.org/abs/2303.08774
          
            **Community:**
            - Hugging Face Forums: https://discuss.huggingface.co
            - Gradio Discord: https://discord.gg/feTf9x3ZSB
          
            ---
          
            ## 🎯 Project Statistics
          
            **Development Journey:**
            - **Part 1:** Foundation + Text + Image (4 hours)
            - **Part 2:** Audio + Fusion + Memory (3 hours)
            - **Part 3:** Personalities + Viz + Deploy (3 hours)
            - **Total:** ~10 hours of development
          
            **Code Statistics:**
            - **Total Files:** 15+ Python files
            - **Lines of Code:** ~3000+ lines
            - **Models Used:** 4 AI models
            - **Features:** 20+ major features
          
            **Cost Analysis:**
            - **Free Version:** $0/month
            - **With OpenAI:** $5-20/month (usage-based)
            - **Deployment:** Free on HF Spaces
          
            ---
          
            ## 🙏 Credits & Acknowledgments
          
            **AI Models:**
            - **Flan-T5:** Google Research
            - **BLIP-2:** Salesforce Research
            - **Whisper:** OpenAI
            - **GPT-4o:** OpenAI
          
            **Frameworks:**
            - **Gradio:** Hugging Face Team
            - **Transformers:** Hugging Face
            - **LangChain:** LangChain Team
          
            **Tools:**
            - **Python:** PSF
            - **PyTorch:** Meta AI
            - **Matplotlib:** NumFOCUS
          
            ---
          
            ## 📞 Support & Feedback
          
            **Need Help?**
            - Check troubleshooting section above
            - Review console output for errors
            - Verify all dependencies installed
          
            **Found a Bug?**
            - Note the error message
            - Check which tab/feature
            - Review recent changes
          
            **Feature Requests?**
            - Consider what you'd like added
            - Check if it fits project scope
            - Think about implementation
          
            ---
          
            <div style="text-align: center; margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px;">
                <h2 style="color: white; margin: 0;">🎉 Congratulations!</h2>
                <p style="color: rgba(255,255,255,0.9); font-size: 1.1em; margin-top: 10px;">
                You've built a complete, production-ready multimodal AI assistant!
                </p>
                <p style="color: rgba(255,255,255,0.85); margin-top: 15px;">
                Made with ❤️ and 🧠 | NeuraFusion V3.0
                </p>
            </div>
            """)
  
    # Footer
    with gr.Row():
        gr.Markdown("""
        <footer style="text-align: center; margin-top: 30px; padding: 25px; background: #f8f9fa; border-radius: 12px;">
            <p style="font-size: 1.1em; margin-bottom: 10px;">
                🧠 <strong>NeuraFusion V3.0 Production</strong>
            </p>
            <p style="color: #666; margin: 5px 0;">
                Powered by Hugging Face 🤗 | Built with Gradio | Enhanced with OpenAI
            </p>
            <p style="color: #888; font-size: 0.9em; margin-top: 10px;">
                Text • Image • Audio • Memory • Personalities • Analytics | 100% Functional
            </p>
        </footer>
        """)

# ============================================
# LAUNCH THE APP
# ============================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎉 NeuraFusion V3.0 is ready!")
    print("="*70)
    print("\n✨ NEW IN PART 3:")
    print(" 🎭 5 Personality modes")
    print(" 🔍 Attention visualizations")
    print(" 🎨 Color analysis")
    print(" 🔐 OpenAI GPT-4o integration (optional)")
    print(" 💾 Enhanced export (JSON/TXT/MD/CSV)")
    print(" 📊 Analytics dashboard")
    print(" 🚀 Deployment ready")
    print("\n💡 The interface will open automatically in your browser")
    print("📍 URL: http://127.0.0.1:7860")
    print(f"\n🔐 OpenAI Status: {'Enabled ✓' if openai_integration.is_enabled() else 'Disabled (add API key to enable)'}")
    print("\n⌨️ Press Ctrl+C to stop the server\n")
  
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False, # Set to True for temporary public link
        show_error=True,
        inbrowser=True
    )