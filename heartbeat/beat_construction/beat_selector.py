import os
import openai
from typing import Dict
from .test import beats, play_beat, play_beat_with_variations  # Import from test.py

class EmotionalBeatSelector:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.beats = beats  # Use beats dictionary from test.py
        self.pattern_cache: Dict[str, str] = {}
        
    def get_beat_for_emotion(self, emotion: str, current_bpm: float) -> str:
        """
        Use ChatGPT to select an appropriate beat name for the detected emotion.
        Returns the name of the beat to play
        """
        # Check cache first
        if emotion in self.pattern_cache:
            return self.pattern_cache[emotion]

        # Prepare prompt for ChatGPT
        prompt = f"""
        Given the following emotional state: '{emotion}' and current BPM: {current_bpm},
        select the most appropriate beat pattern from this list to help normalize the person's emotional state:
        
        Available patterns:
        {list(self.beats.keys())}
        
        Consider these guidelines:
        - For high stress/anxiety: Choose slower, simpler patterns
        - For low energy: Choose upbeat, complex patterns
        - For neutral states: Choose moderate patterns
        
        Return only the exact name of one pattern from the list.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a music therapy expert specializing in drum patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=50
            )
            
            beat_name = response.choices[0].message.content.strip()
            
            # Verify the beat exists
            if beat_name in self.beats:
                # Cache the result
                self.pattern_cache[emotion] = beat_name
                return beat_name
            else:
                return list(self.beats.keys())[0]  # Return first beat as fallback
            
        except Exception as e:
            print(f"Error getting beat recommendation: {str(e)}")
            return list(self.beats.keys())[0]  # Return first beat as fallback

    def play_emotional_beat(self, emotion: str, bpm: float, with_variations: bool = False):
        """
        Select and play a beat based on emotion using the same functions as test.py
        """
        beat_name = self.get_beat_for_emotion(emotion, bpm)
        
        if with_variations:
            play_beat_with_variations(beat_name, bpm)
        else:
            play_beat(beat_name, bpm)
        
        return beat_name
