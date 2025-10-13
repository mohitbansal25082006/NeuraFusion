"""
Memory Manager - Handles conversation history and context
Part 2: NeuraFusion Conversation Memory
"""

from datetime import datetime
from collections import deque
import json

class MemoryManager:
    """
    Manages conversation history and context for NeuraFusion.
    
    Features:
    - Store multi-turn conversations
    - Track different modalities used
    - Provide context for follow-up questions
    - Export conversation history
    - Clear and reset functionality
    """
    
    def __init__(self, max_history=50, max_context_tokens=1000):
        """
        Initialize memory manager.
        
        Args:
            max_history: Maximum number of conversation turns to keep
            max_context_tokens: Approximate max tokens for context window
        """
        self.max_history = max_history
        self.max_context_tokens = max_context_tokens
        
        # Main conversation storage
        # Each entry: {'role': 'user'/'assistant', 'content': str, 'metadata': dict}
        self.conversation_history = deque(maxlen=max_history)
        
        # Session metadata
        self.session_info = {
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'start_time': datetime.now().isoformat(),
            'total_turns': 0,
            'modalities_used': set()
        }
        
        # Current context window (for model input)
        self.current_context = []
        
        print(f"✅ Memory Manager initialized (max history: {max_history})")
    
    def add_message(self, role, content, modalities=None, metadata=None):
        """
        Add a message to conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content (text)
            modalities: List of modalities used (e.g., ['text', 'image'])
            metadata: Additional metadata dictionary
        
        Returns:
            The created message entry
        """
        
        if not content or len(str(content).strip()) == 0:
            return None
        
        # Create message entry
        message = {
            'role': role,
            'content': str(content).strip(),
            'timestamp': datetime.now().isoformat(),
            'modalities': modalities or ['text'],
            'metadata': metadata or {}
        }
        
        # Add to history
        self.conversation_history.append(message)
        
        # Update session info
        self.session_info['total_turns'] += 1
        if modalities:
            self.session_info['modalities_used'].update(modalities)
        
        return message
    
    def add_user_message(self, content, modalities=None, metadata=None):
        """
        Add a user message (convenience method).
        
        Args:
            content: User's message
            modalities: Modalities used
            metadata: Additional info
        
        Returns:
            Message entry
        """
        return self.add_message('user', content, modalities, metadata)
    
    def add_assistant_message(self, content, modalities=None, metadata=None):
        """
        Add an assistant message (convenience method).
        
        Args:
            content: Assistant's response
            modalities: Modalities used
            metadata: Additional info
        
        Returns:
            Message entry
        """
        return self.add_message('assistant', content, modalities, metadata)
    
    def get_recent_history(self, n=10):
        """
        Get the n most recent messages.
        
        Args:
            n: Number of recent messages to retrieve
        
        Returns:
            List of recent messages
        """
        history_list = list(self.conversation_history)
        return history_list[-n:] if len(history_list) > n else history_list
    
    def get_context_for_model(self, max_turns=5):
        """
        Get formatted conversation context for model input.
        
        Args:
            max_turns: Maximum number of recent turns to include
        
        Returns:
            Formatted context string
        """
        recent = self.get_recent_history(n=max_turns * 2)  # *2 because user+assistant = 1 turn
        
        if not recent:
            return ""
        
        # Format as conversation
        context_parts = []
        for msg in recent:
            role_label = "User" if msg['role'] == 'user' else "Assistant"
            context_parts.append(f"{role_label}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def get_gradio_chatbot_history(self):
        """
        Get conversation history in Gradio chatbot format.
        Gradio expects: [[user_msg, bot_msg], [user_msg, bot_msg], ...]
        
        Returns:
            List of [user, bot] message pairs
        """
        history_list = list(self.conversation_history)
        gradio_history = []
        
        # Group messages into pairs
        i = 0
        while i < len(history_list):
            user_msg = None
            bot_msg = None
            
            # Get user message
            if i < len(history_list) and history_list[i]['role'] == 'user':
                user_msg = history_list[i]['content']
                i += 1
            
            # Get assistant message
            if i < len(history_list) and history_list[i]['role'] == 'assistant':
                bot_msg = history_list[i]['content']
                i += 1
            
            # Add pair if we have at least user message
            if user_msg:
                gradio_history.append([user_msg, bot_msg or ""])
        
        return gradio_history
    
    def clear_history(self):
        """
        Clear all conversation history and reset session.
        """
        self.conversation_history.clear()
        self.current_context = []
        
        # Reset session info but keep session ID
        self.session_info['total_turns'] = 0
        self.session_info['modalities_used'] = set()
        
        print("🗑️ Conversation history cleared")
    
    def get_session_summary(self):
        """
        Get a summary of the current session.
        
        Returns:
            Dictionary with session statistics
        """
        history_list = list(self.conversation_history)
        
        # Count messages by role
        user_messages = sum(1 for msg in history_list if msg['role'] == 'user')
        assistant_messages = sum(1 for msg in history_list if msg['role'] == 'assistant')
        
        # Calculate duration
        if history_list:
            start = datetime.fromisoformat(self.session_info['start_time'])
            duration_seconds = (datetime.now() - start).total_seconds()
        else:
            duration_seconds = 0
        
        return {
            'session_id': self.session_info['session_id'],
            'start_time': self.session_info['start_time'],
            'duration_seconds': duration_seconds,
            'total_turns': self.session_info['total_turns'],
            'user_messages': user_messages,
            'assistant_messages': assistant_messages,
            'modalities_used': list(self.session_info['modalities_used']),
            'history_length': len(history_list)
        }
    
    def export_to_json(self, filepath=None):
        """
        Export conversation history to JSON file.
        
        Args:
            filepath: Path to save JSON file (optional)
                     If None, uses session_id as filename
        
        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = f"conversation_{self.session_info['session_id']}.json"
        
        # Prepare export data
        export_data = {
            'session_info': {
                'session_id': self.session_info['session_id'],
                'start_time': self.session_info['start_time'],
                'export_time': datetime.now().isoformat(),
                'total_turns': self.session_info['total_turns'],
                'modalities_used': list(self.session_info['modalities_used'])
            },
            'conversation': list(self.conversation_history)
        }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Conversation exported to: {filepath}")
        return filepath
    
    def export_to_text(self, filepath=None):
        """
        Export conversation history to readable text file.
        
        Args:
            filepath: Path to save text file (optional)
        
        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = f"conversation_{self.session_info['session_id']}.txt"
        
        # Create formatted text
        lines = [
            "="*70,
            "NEURAFUSION CONVERSATION EXPORT",
            "="*70,
            f"Session ID: {self.session_info['session_id']}",
            f"Start Time: {self.session_info['start_time']}",
            f"Export Time: {datetime.now().isoformat()}",
            f"Total Turns: {self.session_info['total_turns']}",
            f"Modalities Used: {', '.join(self.session_info['modalities_used'])}",
            "="*70,
            ""
        ]
        
        # Add conversation
        for i, msg in enumerate(self.conversation_history, 1):
            role_label = "👤 USER" if msg['role'] == 'user' else "🤖 ASSISTANT"
            timestamp = msg['timestamp'].split('T')[1][:8]  # Extract HH:MM:SS
            
            lines.append(f"[{i}] {role_label} ({timestamp})")
            lines.append("-" * 70)
            lines.append(msg['content'])
            lines.append("")
        
        lines.append("="*70)
        lines.append("END OF CONVERSATION")
        lines.append("="*70)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"💾 Conversation exported to: {filepath}")
        return filepath
    
    def import_from_json(self, filepath):
        """
        Import conversation history from JSON file.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Clear current history
            self.clear_history()
            
            # Load session info
            if 'session_info' in data:
                self.session_info.update(data['session_info'])
                if 'modalities_used' in data['session_info']:
                    self.session_info['modalities_used'] = set(data['session_info']['modalities_used'])
            
            # Load conversation
            if 'conversation' in data:
                for msg in data['conversation']:
                    self.conversation_history.append(msg)
            
            print(f"✅ Conversation imported from: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Import failed: {e}")
            return False
    
    def search_history(self, keyword, case_sensitive=False):
        """
        Search conversation history for a keyword.
        
        Args:
            keyword: Search term
            case_sensitive: Whether search should be case-sensitive
        
        Returns:
            List of matching messages
        """
        matches = []
        
        for msg in self.conversation_history:
            content = msg['content']
            search_content = content if case_sensitive else content.lower()
            search_keyword = keyword if case_sensitive else keyword.lower()
            
            if search_keyword in search_content:
                matches.append(msg)
        
        return matches
    
    def get_modality_statistics(self):
        """
        Get statistics about modality usage in conversation.
        
        Returns:
            Dictionary with modality counts
        """
        modality_counts = {
            'text': 0,
            'image': 0,
            'audio': 0,
            'multimodal': 0
        }
        
        for msg in self.conversation_history:
            modalities = msg.get('modalities', ['text'])
            
            if len(modalities) > 1:
                modality_counts['multimodal'] += 1
            
            for mod in modalities:
                if mod in modality_counts:
                    modality_counts[mod] += 1
        
        return modality_counts
    
    def __len__(self):
        """
        Get number of messages in history.
        """
        return len(self.conversation_history)
    
    def __str__(self):
        """
        String representation of memory manager.
        """
        summary = self.get_session_summary()
        return (
            f"MemoryManager(session={summary['session_id']}, "
            f"messages={summary['history_length']}, "
            f"turns={summary['total_turns']})"
        )


# Test function
if __name__ == "__main__":
    print("🧪 Testing Memory Manager...")
    print("="*70)
    
    # Create memory manager
    memory = MemoryManager(max_history=50)
    
    print("\n" + "="*70)
    print("Test 1: Adding Messages")
    print("="*70)
    
    # Simulate a conversation
    memory.add_user_message("Hello! Can you help me?", modalities=['text'])
    memory.add_assistant_message("Of course! I'm here to help. What do you need?", modalities=['text'])
    
    memory.add_user_message("What's in this image?", modalities=['text', 'image'])
    memory.add_assistant_message("The image shows a beautiful sunset over the ocean.", modalities=['text', 'image'])
    
    memory.add_user_message("Tell me more about sunsets", modalities=['audio'])
    memory.add_assistant_message("Sunsets occur when the sun descends below the horizon...", modalities=['text', 'audio'])
    
    print(f"✅ Added 6 messages to history")
    print(f"📊 History length: {len(memory)}")
    
    print("\n" + "="*70)
    print("Test 2: Retrieving History")
    print("="*70)
    
    recent = memory.get_recent_history(n=4)
    print(f"Recent messages (last 4):")
    for msg in recent:
        print(f"  - {msg['role']}: {msg['content'][:50]}...")
    
    print("\n" + "="*70)
    print("Test 3: Gradio Format")
    print("="*70)
    
    gradio_hist = memory.get_gradio_chatbot_history()
    print(f"Gradio chat history (pairs): {len(gradio_hist)}")
    for i, pair in enumerate(gradio_hist, 1):
        print(f"  Turn {i}: User={pair[0][:30]}... | Bot={pair[1][:30]}...")
    
    print("\n" + "="*70)
    print("Test 4: Session Summary")
    print("="*70)
    
    summary = memory.get_session_summary()
    print(f"Session ID: {summary['session_id']}")
    print(f"Total turns: {summary['total_turns']}")
    print(f"User messages: {summary['user_messages']}")
    print(f"Assistant messages: {summary['assistant_messages']}")
    print(f"Modalities used: {', '.join(summary['modalities_used'])}")
    
    print("\n" + "="*70)
    print("Test 5: Modality Statistics")
    print("="*70)
    
    mod_stats = memory.get_modality_statistics()
    for modality, count in mod_stats.items():
        print(f"  {modality}: {count} occurrences")
    
    print("\n" + "="*70)
    print("Test 6: Search History")
    print("="*70)
    
    matches = memory.search_history("sunset")
    print(f"Found {len(matches)} messages containing 'sunset':")
    for match in matches:
        print(f"  - {match['role']}: {match['content'][:60]}...")
    
    print("\n" + "="*70)
    print("Test 7: Export")
    print("="*70)
    
    json_file = memory.export_to_json("test_conversation.json")
    txt_file = memory.export_to_text("test_conversation.txt")
    
    print(f"✅ Exported to JSON: {json_file}")
    print(f"✅ Exported to TXT: {txt_file}")
    
    print("\n" + "="*70)
    print("Test 8: Context for Model")
    print("="*70)
    
    context = memory.get_context_for_model(max_turns=2)
    print("Context string (last 2 turns):")
    print(context)
    
    print("\n" + "="*70)
    print("✅ All Memory Manager tests complete!")
    print("="*70)
    print(f"\n{memory}")