"""
Sports Commentary Engine
Generates AI-powered audio commentary for real-time match actions.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

# from google.cloud import texttospeech  # Temporarily disabled - will use XTTS v2 instead

# Load environment variables from a .env file (if present)
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CommentaryEngine:
    """
    A modular engine for generating AI-powered sports commentary with audio synthesis.
    
    This class processes match actions and converts them into engaging audio commentary
    using Google's Generative AI (Gemini) and Text-to-Speech services.
    """
    
    def __init__(
        self,
        output_dir: str = "commentary_output",
        voice_name: str = "tr-TR-Wavenet-D",
        language_code: str = "tr-TR",
        model_name: str = "models/gemini-2.0-flash-lite"
    ):
        """
        Initialize the Commentary Engine.
        
        Args:
            output_dir: Directory to save audio files and metadata
            voice_name: Google Cloud TTS voice identifier
            language_code: Language code for TTS
            model_name: Gemini model to use (default: gemini-1.5-flash for free tier)
        
        Raises:
            ValueError: If required API keys are not set in environment variables
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.voice_name = voice_name
        self.language_code = language_code
        self.model_name = model_name
        
        # Initialize API clients
        self._initialize_genai()
        self._initialize_tts()
        
        # Storage for generated commentary
        self.commentary_history = []
        
    def _initialize_genai(self) -> None:
        """Initialize Google Generative AI client."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Please set it before using the Commentary Engine."
            )

        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to configure Generative AI client: {e}")
            raise

        # Try to initialize the requested model; if unavailable, list models and pick a fallback
        try:
            self.llm_model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Model '{self.model_name}' init failed: {e}. Attempting to list available models.")
            try:
                models = genai.list_models()
                names = []
                for m in models:
                    if isinstance(m, dict):
                        names.append(m.get("name"))
                    else:
                        names.append(getattr(m, "name", str(m)))

                logger.info(f"Available models: {names}")

                if not names:
                    raise RuntimeError("No models available from list_models()")

                # Choose first available model as fallback
                self.model_name = names[0]
                self.llm_model = genai.GenerativeModel(self.model_name)
                logger.info(f"Switched to available model: {self.model_name}")
            except Exception as e2:
                logger.error(f"Failed to list/switch models: {e2}")
                raise
    
    def _initialize_tts(self) -> None:
        """
        Initialize Text-to-Speech client.
        Currently mocked - will be replaced with XTTS v2 implementation.
        """
        # TODO: Implement XTTS v2 initialization
        # For now, TTS is disabled to save disk space during LLM testing
        self.tts_client = None
        logger.info("TTS client initialized (mocked - XTTS v2 pending)")
    
    def _create_commentary_prompt(self, match_data: Dict) -> str:
        """
        Create a prompt for the LLM based on match data.
        
        Args:
            match_data: Dictionary containing match information
            
        Returns:
            Formatted prompt string for the LLM
        """
        team_a = match_data.get("team_a", "Team A")
        team_b = match_data.get("team_b", "Team B")
        active_player = match_data.get("active_player", "a player")
        action_type = match_data.get("action_type", "action")
        emotion = match_data.get("emotion", "excited")
        referee_side = match_data.get("referee_side", "")
        
        prompt = f"""You are an enthusiastic football commentator providing live match commentary.
        
Match Context:
- Teams: {team_a} vs {team_b}
- Active Player: {active_player}
- Action: {action_type}
- Emotion Level: {emotion}
{f"- Referee Decision: {referee_side}" if referee_side else ""}

Generate a short, engaging commentary (max 2 sentences) that captures the excitement of this moment.
Use dynamic language, emotion, and make it sound natural for live sports broadcasting.
Keep it concise and impactful."""
        
        return prompt
    
    def _generate_commentary_text(self, match_data: Dict) -> Optional[str]:
        """
        Generate commentary text using Gemini LLM.
        
        Args:
            match_data: Dictionary containing match information
            
        Returns:
            Generated commentary text or None if generation fails
        """
        try:
            prompt = self._create_commentary_prompt(match_data)

            # Generate content with safety settings
            response = self.llm_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=150,
                    temperature=0.9,  # Higher temperature for more creative commentary
                )
            )

            commentary_text = response.text.strip()
            logger.info(f"Generated commentary: {commentary_text[:50]}...")
            return commentary_text

        except Exception as e:
            logger.error(f"Error generating commentary text: {e}")
            # If LLM fails, return None so caller can treat it as an error
            return None
    
    def _synthesize_audio(self, text: str, timestamp: str) -> Optional[str]:
        """
        Convert text to speech using XTTS v2.
        Currently mocked to save disk space during LLM testing.
        
        Args:
            text: Commentary text to synthesize
            timestamp: Timestamp for file naming
            
        Returns:
            Path to the generated audio file (mocked) or None if synthesis fails
        """
        try:
            # TODO: Implement XTTS v2 synthesis here
            # For now, return a mock path without actually creating the file
            
            safe_timestamp = timestamp.replace(":", "-").replace(" ", "_")
            audio_filename = f"commentary_{safe_timestamp}.mp3"
            audio_path = self.output_dir / audio_filename
            
            logger.info(f"[MOCKED] Audio would be saved to: {audio_path}")
            logger.info(f"[MOCKED] Text to synthesize: {text[:100]}...")
            
            # Return the path as if the file was created (for testing purposes)
            return str(audio_path)
            
        except Exception as e:
            logger.error(f"Error in audio synthesis mock: {e}")
            return None
    
    def process_match_action(self, match_data: Dict) -> Dict:
        """
        Process a match action and generate commentary with audio.
        
        Args:
            match_data: Dictionary containing:
                - team_a: Name of team A
                - team_b: Name of team B
                - active_player: Name of the player performing the action
                - action_type: Type of action (goal, pass, tackle, etc.)
                - emotion: Emotion level (excited, tense, calm, etc.)
                - timestamp: Timestamp of the action
                - referee_side: Optional referee decision
        
        Returns:
            Dictionary containing:
                - text: Generated commentary text
                - audio_path: Path to audio file
                - timestamp: Original timestamp
                - status: Success or error status
        """
        timestamp = match_data.get("timestamp", datetime.now().isoformat())
        
        try:
            # Generate commentary text
            commentary_text = self._generate_commentary_text(match_data)
            
            if not commentary_text:
                return {
                    "text": None,
                    "audio_path": None,
                    "timestamp": timestamp,
                    "status": "error",
                    "error": "Failed to generate commentary text"
                }
            
            # Synthesize audio
            audio_path = self._synthesize_audio(commentary_text, timestamp)
            
            # Prepare result
            result = {
                "text": commentary_text,
                "audio_path": audio_path,
                "timestamp": timestamp,
                "status": "success" if audio_path else "partial_success",
                "match_data": match_data
            }
            
            # Store in history
            self.commentary_history.append(result)
            
            # Save metadata
            self._save_metadata(result)
            
            logger.info(f"Successfully processed match action at {timestamp}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing match action: {e}")
            return {
                "text": None,
                "audio_path": None,
                "timestamp": timestamp,
                "status": "error",
                "error": str(e)
            }
    
    def _save_metadata(self, result: Dict) -> None:
        """
        Save commentary metadata to JSON file.
        
        Args:
            result: Result dictionary from process_match_action
        """
        try:
            metadata_file = self.output_dir / "commentary_metadata.json"
            
            # Load existing metadata
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                metadata = []
            
            # Append new result
            metadata.append(result)
            
            # Save updated metadata
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info("Metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def get_commentary_by_timestamp(self, timestamp: str) -> Optional[Dict]:
        """
        Retrieve commentary data by timestamp.
        
        Args:
            timestamp: Timestamp to search for
            
        Returns:
            Commentary data or None if not found
        """
        for commentary in self.commentary_history:
            if commentary.get("timestamp") == timestamp:
                return commentary
        return None
    
    def get_all_commentary(self) -> list:
        """
        Get all generated commentary.
        
        Returns:
            List of all commentary results
        """
        return self.commentary_history
    
    def clear_history(self) -> None:
        """Clear commentary history from memory."""
        self.commentary_history = []
        logger.info("Commentary history cleared")


# Sample usage
if __name__ == "__main__":
    # Example: Initialize the engine
    print("=" * 60)
    print("Sports Commentary Engine - Demo")
    print("=" * 60)
    print()
    
    # Check if API keys are set
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set!")
        print("Please set it using: export GOOGLE_API_KEY='your-api-key'")
        exit(1)
    
    try:
        # Initialize engine with Turkish voice
        engine = CommentaryEngine(
            output_dir="commentary_output",
            voice_name="tr-TR-Wavenet-D",
            language_code="tr-TR",
            model_name="models/gemma-3-1b-it"
        )
        
        print("✓ Commentary Engine initialized successfully\n")
        
        # Sample match action 1: Goal
        print("Processing Match Action 1: Goal")
        print("-" * 60)
        match_action_1 = {
            "team_a": "Galatasaray",
            "team_b": "Fenerbahçe",
            "active_player": "Icardi",
            "action_type": "goal",
            "emotion": "ecstatic",
            "timestamp": "2026-02-04T15:23:45",
            "referee_side": ""
        }
        
        result_1 = engine.process_match_action(match_action_1)
        print(f"Status: {result_1['status']}")
        print(f"Text: {result_1['text']}")
        print(f"Audio: {result_1['audio_path']}")
        print()
        
        # Sample match action 2: Penalty decision
        print("Processing Match Action 2: Penalty")
        print("-" * 60)
        match_action_2 = {
            "team_a": "Galatasaray",
            "team_b": "Fenerbahçe",
            "active_player": "referee",
            "action_type": "penalty decision",
            "emotion": "tense",
            "timestamp": "2026-02-04T15:45:12",
            "referee_side": "Galatasaray"
        }
        
        result_2 = engine.process_match_action(match_action_2)
        print(f"Status: {result_2['status']}")
        print(f"Text: {result_2['text']}")
        print(f"Audio: {result_2['audio_path']}")
        print()
        
        # Sample match action 3: Dramatic save
        print("Processing Match Action 3: Save")
        print("-" * 60)
        match_action_3 = {
            "team_a": "Galatasaray",
            "team_b": "Fenerbahçe",
            "active_player": "Muslera",
            "action_type": "incredible save",
            "emotion": "thrilling",
            "timestamp": "2026-02-04T15:52:30",
            "referee_side": ""
        }
        
        result_3 = engine.process_match_action(match_action_3)
        print(f"Status: {result_3['status']}")
        print(f"Text: {result_3['text']}")
        print(f"Audio: {result_3['audio_path']}")
        print()
        
        # Display summary
        print("=" * 60)
        print(f"Total commentary generated: {len(engine.get_all_commentary())}")
        print(f"Output directory: {engine.output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
