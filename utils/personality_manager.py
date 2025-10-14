"""
Personality Manager - Handles different AI personality modes
Part 3: NeuraFusion Personality System
"""

import json
from pathlib import Path

class PersonalityManager:
    """
    Manages different AI personality modes for varied interaction styles.
    
    Personalities affect:
    - Response tone and style
    - Temperature (creativity level)
    - Response length
    - Prompt framing
    """
    
    def __init__(self, config_path="config/personalities.json"):
        """
        Initialize personality manager.
        
        Args:
            config_path: Path to personalities configuration JSON
        """
        self.config_path = Path(config_path)
        self.personalities = {}
        self.current_personality = "friend"  # Default
        
        self.load_personalities()
        print(f"✅ Personality Manager initialized ({len(self.personalities)} modes)")
    
    def load_personalities(self):
        """Load personality configurations from JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.personalities = json.load(f)
            
            # Validate that we have at least one personality
            if not self.personalities:
                raise ValueError("No personalities found in config")
            
            # Set default if current doesn't exist
            if self.current_personality not in self.personalities:
                self.current_personality = list(self.personalities.keys())[0]
                
        except FileNotFoundError:
            print(f"⚠️ Personalities config not found: {self.config_path}")
            print("💡 Creating default personalities...")
            self._create_default_personalities()
        except Exception as e:
            print(f"❌ Error loading personalities: {e}")
            self._create_default_personalities()
    
    def _create_default_personalities(self):
        """Create default personality configurations."""
        self.personalities = {
            "friend": {
                "name": "Friend Mode",
                "emoji": "😊",
                "system_prompt": "You are a friendly assistant. Be warm and conversational.",
                "temperature": 0.8,
                "max_length": 400
            },
            "professional": {
                "name": "Professional Mode",
                "emoji": "💼",
                "system_prompt": "You are a professional assistant. Be concise and formal.",
                "temperature": 0.6,
                "max_length": 500
            }
        }
        self.current_personality = "friend"
    
    def set_personality(self, personality_key):
        """
        Switch to a different personality mode.
        
        Args:
            personality_key: Key of personality to activate
        
        Returns:
            Success message or error string
        """
        if personality_key not in self.personalities:
            available = ', '.join(self.personalities.keys())
            return f"❌ Unknown personality. Available: {available}"
        
        self.current_personality = personality_key
        config = self.personalities[personality_key]
        return f"✅ Switched to {config['emoji']} {config['name']}"
    
    def get_current_personality(self):
        """
        Get current personality configuration.
        
        Returns:
            Dictionary with personality settings
        """
        return self.personalities.get(self.current_personality, {})
    
    def get_system_prompt(self):
        """
        Get system prompt for current personality.
        
        Returns:
            System prompt string
        """
        config = self.get_current_personality()
        return config.get('system_prompt', '')
    
    def get_temperature(self):
        """
        Get temperature setting for current personality.
        
        Returns:
            Temperature float (0.0-1.0)
        """
        config = self.get_current_personality()
        return config.get('temperature', 0.7)
    
    def get_max_length(self):
        """
        Get max response length for current personality.
        
        Returns:
            Max length integer
        """
        config = self.get_current_personality()
        return config.get('max_length', 500)
    
    def format_prompt_with_personality(self, user_input):
        """
        Format user input with personality context.
        
        Args:
            user_input: Raw user message
        
        Returns:
            Enhanced prompt string
        """
        system_prompt = self.get_system_prompt()
        
        # Create contextual prompt
        enhanced_prompt = f"{system_prompt}\n\nUser: {user_input}\n\nAssistant:"
        
        return enhanced_prompt
    
    def get_all_personalities(self):
        """
        Get list of all available personalities.
        
        Returns:
            List of personality info dictionaries
        """
        result = []
        for key, config in self.personalities.items():
            result.append({
                'key': key,
                'name': config.get('name', key),
                'emoji': config.get('emoji', '🤖'),
                'is_active': key == self.current_personality
            })
        return result
    
    def get_personality_description(self, personality_key=None):
        """
        Get detailed description of a personality.
        
        Args:
            personality_key: Personality to describe (None = current)
        
        Returns:
            Formatted description string
        """
        key = personality_key or self.current_personality
        
        if key not in self.personalities:
            return "Personality not found"
        
        config = self.personalities[key]
        
        description = [
            f"{config.get('emoji', '🤖')} **{config.get('name', key)}**",
            "",
            f"**Style:** {config.get('system_prompt', 'N/A')[:100]}...",
            "",
            f"**Settings:**",
            f"- Temperature: {config.get('temperature', 0.7)}",
            f"- Max Length: {config.get('max_length', 500)} tokens"
        ]
        
        if 'style_traits' in config:
            description.append("")
            description.append("**Traits:**")
            for trait in config['style_traits']:
                description.append(f"- {trait}")
        
        return "\n".join(description)
    
    def export_config(self, filepath=None):
        """
        Export current personality configurations.
        
        Args:
            filepath: Where to save (optional)
        
        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = "personalities_backup.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.personalities, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def __str__(self):
        """String representation."""
        config = self.get_current_personality()
        return f"PersonalityManager(current='{self.current_personality}' {config.get('emoji', '')})"


# Test function
if __name__ == "__main__":
    print("🧪 Testing Personality Manager...")
    print("="*70)
    
    manager = PersonalityManager()
    
    print("\n" + "="*70)
    print("Test 1: List All Personalities")
    print("="*70)
    
    personalities = manager.get_all_personalities()
    for p in personalities:
        active = "✓" if p['is_active'] else " "
        print(f"[{active}] {p['emoji']} {p['name']} ({p['key']})")
    
    print("\n" + "="*70)
    print("Test 2: Switch Personalities")
    print("="*70)
    
    for key in ['mentor', 'analyst', 'friend']:
        result = manager.set_personality(key)
        print(result)
        print(f"  Temperature: {manager.get_temperature()}")
        print(f"  Max Length: {manager.get_max_length()}")
    
    print("\n" + "="*70)
    print("Test 3: Format Prompts")
    print("="*70)
    
    test_input = "Explain quantum computing"
    
    for key in ['mentor', 'professional']:
        manager.set_personality(key)
        prompt = manager.format_prompt_with_personality(test_input)
        print(f"\n{key.upper()}:")
        print(prompt[:200] + "...")
    
    print("\n" + "="*70)
    print("Test 4: Get Description")
    print("="*70)
    
    desc = manager.get_personality_description('analyst')
    print(desc)
    
    print("\n" + "="*70)
    print("✅ All Personality Manager tests complete!")
    print("="*70)