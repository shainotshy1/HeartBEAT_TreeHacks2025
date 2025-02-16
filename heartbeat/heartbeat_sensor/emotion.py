from enum import Enum

class Emotion(Enum):
    HIGH_STRESS_FEAR = "High Stress/Fear"
    ANXIOUS = "Anxious"
    DEEP_RELAXATION = "Deep Relaxation"
    CALM = "Calm"
    HAPPY_EXCITED = "Happy/Excited"
    NEUTRAL = "Neutral"
    MILD_AROUSAL = "Mild Arousal"
    MILD_RELAXATION = "Mild Relaxation"
    MILD_STRESS = "Mild Stress"
    FOCUS_CONCENTRATION = "Focus/Concentration"
    INCREASED_PARASYMPATHETIC_ACTIVITY = "Increased Parasympathetic Activity"
    INCREASED_SYMPATHETIC_ACTIVITY = "Decreased Sympathetic Activity"
    MIXED_EMOTIONAL_STATE = "Mixed Emotional State"
    
all_emotion_str = [e.value for e in Emotion]