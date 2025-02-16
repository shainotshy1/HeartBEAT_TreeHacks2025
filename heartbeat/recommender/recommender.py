import os
import numpy as np
import pandas as pd
import time
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List

from heartbeat.heartbeat_sensor.emotion import Emotion, all_emotion_str
from heartbeat.beat_construction.beat_constructor import TimeSignature

np.random.seed(42)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_url = 'https://api.groq.com/openai/v1/chat/completions'
groq_api_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {groq_api_key}",
}

temperature = 0.2

def sample_list(l: List, n: int):
    idx = np.random.choice(len(l), n, replace=False)
    return [l[i] for i in idx]

def get_patterns(max_samples: int = 20):
    drum_patterns = pd.read_csv('heartbeat/beat_construction/drum_patterns.csv')
    synth_patterns = pd.read_csv('heartbeat/beat_construction/synth_patterns.csv')
    drum_patterns_str = drum_patterns['label'].astype(str).tolist()
    synth_patterns_str = synth_patterns['label'].astype(str).tolist()
    return (
        sample_list(drum_patterns_str, max_samples), 
        sample_list(synth_patterns_str, max_samples)
    )

def get_drum_sample_classes(max_samples: int = 20):
    with open('data/drum_classes.txt', 'r') as f:
        sample_classes = f.read().splitlines()
    return sample_list(sample_classes, max_samples)

def get_drum_samples_data():
    return pd.read_csv('data/drum_full.csv')

def get_drum_samples_filenames(drum_samples_data: pd.DataFrame,
                               sample_class: str, 
                               max_samples: int = 20):
    drum_sample_filenames = drum_samples_data.query(f'drum_class == "{sample_class}"')['file'].to_list()
    return sample_list(drum_sample_filenames, max_samples)

def get_expected_format_prompt():
    return 'You may think briefly to come to a conclusion (no more than one sentence); we are just looking for an educated guess. Put your final answer on a new line at the end of your response, directly copied from the input.'

def get_system_prompt():
    return "You are an expert beat producer, and are very good at guessing beat patterns and samples based on emotions and filenames."

def get_drum_pattern_prompt(emotion: str, drum_patterns: List[str]):
    return f"Given that the emotion is {emotion}, guess the best-matching drum pattern out of the following list. {get_expected_format_prompt()} \n{', '.join(drum_patterns)}"

def get_drum_sample_class_prompt(emotion: str, drum_sample_classes: List[str]):
    return f"Given that the emotion is {emotion}, guess the best-matching drum sample out of the following list of drum sample classes. {get_expected_format_prompt()} \n{', '.join(drum_sample_classes)}"

def get_drum_sample_filename_prompt(emotion: str, drum_sample_class: str, drum_sample_filenames: List[str]):
    return f"Given that the emotion is {emotion} and the sample class is {drum_sample_class}, guess the best-matching drum sample out of the following list of .wav filenames. {get_expected_format_prompt()} \n{', '.join(drum_sample_filenames)}"


def query_openai(emotion: str, 
                 time_signature: TimeSignature,
                 curr_layer_patterns_so_far: List[str],
                 curr_layer_samples_so_far: List[str],
                 prev_layers_so_far: List[str],
                 verbose=False):
    """emotion -> pattern, sample"""
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    get_content = lambda completion: completion.choices[0].message.content
    extract_answer = lambda content: content.split('\n')[-1]
    
    system_prompt = {"role": "system", "content": get_system_prompt()}
    drum_pattern_prompt = {"role": "user", "content": get_drum_pattern_prompt(emotion, drum_patterns)}
    # synth_prompt = {"role": "user", "content": f"Given that the emotion is {emotion}, guess the best-matching synth pattern out of the following list. \n{', '.join(synth_patterns)}"}
    drum_sample_class_prompt = {"role": "user", "content": get_drum_sample_class_prompt(emotion, drum_sample_classes)}
    
    t0 = time.time()
    drum_pattern_completion = get_content(openai_client.chat.completions.create(model="gpt-4o-mini", store=True, messages=[system_prompt, drum_pattern_prompt], temperature=temperature))
    drum_pattern = extract_answer(drum_pattern_completion)
    t1 = time.time()
    if verbose:
        print('Drum pattern prompt:', drum_pattern_prompt['content'])
        print()
        print('Drum pattern response:', drum_pattern_completion)
        print('Time (s):', t1 - t0)
        print()
    
    t0 = time.time()
    drum_sample_class_completion = get_content(openai_client.chat.completions.create(model="gpt-4o-mini", store=True, messages=[system_prompt, drum_sample_class_prompt], temperature=temperature))
    drum_sample_class = extract_answer(drum_sample_class_completion)
    t1 = time.time()
    if verbose:
        print('Drum sample class prompt:', drum_sample_class_prompt['content'])
        print()
        print('Drum sample class response:', drum_sample_class_completion)
        print('Time (s):', t1 - t0)
        print()
    
    t0 = time.time()
    drum_sample_filenames = get_drum_samples_filenames(drum_samples_data, drum_sample_class)
    drum_sample_filename_prompt = {"role": "user", "content": get_drum_sample_filename_prompt(emotion, drum_sample_class, drum_sample_filenames)}
    drum_sample_filename_completion = get_content(openai_client.chat.completions.create(model="gpt-4o-mini", store=True, messages=[system_prompt, drum_sample_filename_prompt], temperature=temperature))
    drum_sample_filename = extract_answer(drum_sample_filename_completion)
    t1 = time.time()
    if verbose:
        print('Drum sample filename prompt:', drum_sample_filename_prompt['content'])
        print()
        print('Drum sample filename response:', drum_sample_filename_completion)
        print('Time (s):', t1 - t0)
        print()
    
    return {
        'drum_pattern': drum_pattern,
        'drum_sample_class': drum_sample_class,
        'drum_sample_filename': drum_sample_filename
    }


def query_groq(emotion: str, drum_patterns: List[str], synth_patterns: List[str],
                 drum_sample_classes: List[str], drum_samples_data: pd.DataFrame,
                 verbose=False):
    """emotion -> pattern, sample"""
    if groq_api_key is None:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    def sample_groq(system_prompt, user_prompt):
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
        }
        
        response = requests.post(groq_api_url, headers=groq_api_headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    
    extract_answer = lambda content: content.split('\n')[-1]
    
    system_prompt = get_system_prompt()
    drum_pattern_prompt = get_drum_pattern_prompt(emotion, drum_patterns)
    # synth_prompt = {"role": "user", "content": f"Given that the emotion is {emotion}, guess the best-matching synth pattern out of the following list. \n{', '.join(synth_patterns)}"}
    drum_sample_class_prompt = get_drum_sample_class_prompt(emotion, drum_sample_classes)
    
    t0 = time.time()
    drum_pattern_completion = sample_groq(system_prompt, drum_pattern_prompt)
    drum_pattern = extract_answer(drum_pattern_completion)
    t1 = time.time()
    if verbose:
        print('Drum pattern prompt:', drum_pattern_prompt)
        print()
        print('Drum pattern response:', drum_pattern_completion)
        print('Time (s):', t1 - t0)
        print()
    
    t0 = time.time()
    drum_sample_class_completion = sample_groq(system_prompt, drum_sample_class_prompt)
    drum_sample_class = extract_answer(drum_sample_class_completion)
    t1 = time.time()
    if verbose:
        print('Drum sample class prompt:', drum_sample_class_prompt)
        print()
        print('Drum sample class response:', drum_sample_class_completion)
        print('Time (s):', t1 - t0)
        print()
    
    t0 = time.time()
    drum_sample_filenames = get_drum_samples_filenames(drum_samples_data, drum_sample_class)
    drum_sample_filename_prompt = get_drum_sample_filename_prompt(emotion, drum_sample_class, drum_sample_filenames)
    drum_sample_filename_completion = sample_groq(system_prompt, drum_sample_filename_prompt)
    drum_sample_filename = extract_answer(drum_sample_filename_completion)
    t1 = time.time()
    if verbose:
        print('Drum sample filename prompt:', drum_sample_filename_prompt)
        print()
        print('Drum sample filename response:', drum_sample_filename_completion)
        print('Time (s):', t1 - t0)
        print()
    
    return {
        'drum_pattern': drum_pattern,
        'drum_sample_class': drum_sample_class,
        'drum_sample_filename': drum_sample_filename
    }
    

if __name__ == "__main__":
    drum_patterns, synth_patterns = get_patterns()
    drum_sample_classes = get_drum_sample_classes()
    drum_samples_data = get_drum_samples_data()
    random_emotion = np.random.choice(all_emotion_str)
    print(f'Emotion: {random_emotion}')
    # print(query_openai(random_emotion, drum_patterns, synth_patterns, drum_sample_classes, drum_samples_data, verbose=True))
    print(query_groq(random_emotion, drum_patterns, synth_patterns, drum_sample_classes, drum_samples_data, verbose=True))