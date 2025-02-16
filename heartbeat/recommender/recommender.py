import os
import numpy as np
import pandas as pd
import time
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
import requests
from requests.exceptions import HTTPError
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from heartbeat.heartbeat_sensor.emotion import all_emotion_str

np.random.seed(42)

load_dotenv()

# OpenAI configuration
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI()

# Groq configuration
groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_url = 'https://api.groq.com/openai/v1/chat/completions'
groq_api_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {groq_api_key}",
}

# HuggingFace configuration
hf_model_name = "microsoft/phi-2"  # Smaller, faster model
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token is None:
    raise ValueError("HUGGINGFACE_TOKEN environment variable not set")

# Initialize HuggingFace model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_model_name, token=hf_token, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    hf_model_name, 
    token=hf_token,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

temperature = 0.2

def sample_list(l: List, n: int = 20):
    if len(l) <= n:
        return l
    idx = np.random.choice(len(l), n, replace=False)
    return [l[i] for i in idx]

def get_patterns(max_samples: int = 20):
    patterns = pd.read_csv('heartbeat/beat_construction/patterns.csv')
    synth_patterns = pd.read_csv('heartbeat/beat_construction/synth_patterns.csv')
    patterns_str = patterns['label'].astype(str).tolist()
    synth_patterns_str = synth_patterns['label'].astype(str).tolist()
    return (
        sample_list(patterns_str, max_samples), 
        sample_list(synth_patterns_str, max_samples)
    )

def get_track_groups(max_samples: int = 20):
    with open('data/classes.txt', 'r') as f:
        track_groups = f.read().splitlines()
    return sample_list(track_groups, max_samples)

def get_samples_data():
    return pd.read_csv('data/full.csv')

def get_samples_filenames(samples_data: pd.DataFrame,
                               track_group: str, 
                               max_samples: int = 20):
    sample_filenames = samples_data.query(f'class == "{track_group}"')['file'].to_list()
    return sample_list(sample_filenames, max_samples)

def get_expected_format_prompt():
    return 'You may think briefly to come to a conclusion (no more than one sentence); we are just looking for an educated guess. Put your final answer on a new line at the end of your response, directly copied from the input. You must minimize repitition while ensuring continuity in the style of the beat. Patterns on the current layer should differ from tracks on previous layers.'

def get_system_prompt():
    return "You are an expert beat producer, and are very good at guessing beat patterns and samples based on emotions and filenames."

def get_pattern_prompt(
    emotion: str, 
    all_patterns: List[str],
    curr_layer_patterns_so_far: List[str],
    prev_layers_so_far: List[str]
):
    all_patterns = sample_list(all_patterns)
    recent_patterns = curr_layer_patterns_so_far
    patterns_concat_str = ', '.join(all_patterns)
    return f"Given that the emotion is {emotion}, and that we used the following recent patterns: [{recent_patterns}], and that the following progress towards track layers has already been done: [{prev_layers_so_far}], guess the best-matching drum pattern out of the following list. {get_expected_format_prompt()} \n{patterns_concat_str}"

def get_track_prompt(
    emotion: str, 
    all_tracks: List[str],
    curr_layer_samples_so_far: List[str],
    prev_layers_so_far: List[str]
):
    all_tracks = sample_list(all_tracks)
    recent_tracks = curr_layer_samples_so_far
    tracks_concat_str = '\n'.join(all_tracks)
    return f"Given that the emotion is {emotion}, and that we used the following recent tracks: [{recent_tracks}], and that the following progress towards track layers has already been done: [{prev_layers_so_far}], guess the best-matching drum sample out of the following list. {get_expected_format_prompt()} \n{tracks_concat_str}"

def query_openai(
    all_patterns: List[str],
    all_tracks: Dict[str, List[str]],
    time_signature,  # unused
    curr_layer_patterns_so_far: List[str],
    curr_layer_samples_so_far: List[str],
    prev_layers_so_far,  # unused
    emotion: str,
    verbose=True
):
    """
    Emotion -> pattern, sample
    Each query takes ~0.6 seconds
    """
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not set")
        
    def sample_openai(system_prompt, user_prompt):
        system_prompt = {"role": "system", "content": system_prompt}
        user_prompt = {"role": "user", "content": user_prompt}        
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini", store=True, messages=[system_prompt, user_prompt], temperature=temperature
        )
        completion = completion.choices[0].message.content
        return completion
    
    extract_answer = lambda content: content.split('\n')[-1]
    
    system_prompt = get_system_prompt()
    pattern_prompt = get_pattern_prompt(emotion, all_patterns, curr_layer_patterns_so_far, prev_layers_so_far)
    track_prompt = get_track_prompt(emotion, all_tracks, curr_layer_samples_so_far, prev_layers_so_far)
    
    t0 = time.time()
    pattern_completion = sample_openai(system_prompt, pattern_prompt)
    pattern = extract_answer(pattern_completion)
    t1 = time.time()
    if verbose:
        print('Pattern prompt:', pattern_prompt)
        print()
        print('Pattern response:', pattern_completion)
        print('Time (s):', t1 - t0)
        print()
    
    t0 = time.time()
    track_completion = sample_openai(system_prompt, track_prompt)
    track = extract_answer(track_completion)
    t1 = time.time()
    if verbose:
        print('Sample prompt:', track_prompt)
        print()
        print('Sample response:', track_completion)
        print('Time (s):', t1 - t0)
        print()
        
    return pattern, track

def query_groq(
    all_patterns: List[str],
    all_tracks: Dict[str, List[str]],
    time_signature,  # unused
    curr_layer_patterns_so_far: List[str],
    curr_layer_samples_so_far: List[str],
    prev_layers_so_far,  # unused
    emotion: str,
    verbose=True
):
    """
    Emotion -> pattern, sample
    Each query takes ~0.6 seconds
    """
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
        
        max_tries = 5
        for i in range(max_tries):
            try:
                response = requests.post(groq_api_url, headers=groq_api_headers, data=json.dumps(data))
                response.raise_for_status()
                break
            except HTTPError as http_err:
                if response.status_code != 429:
                    print(http_err)
                    exit(-1)
        if response.status_code == 429:
            print(response)
            exit(-1)
            
        return response.json()['choices'][0]['message']['content']
    
    extract_answer = lambda content: content.split('\n')[-1]
    
    system_prompt = get_system_prompt()
    pattern_prompt = get_pattern_prompt(emotion, all_patterns, curr_layer_patterns_so_far, prev_layers_so_far)
    track_prompt = get_track_prompt(emotion, all_tracks, curr_layer_samples_so_far, prev_layers_so_far)
    
    t0 = time.time()
    pattern_completion = sample_groq(system_prompt, pattern_prompt)
    pattern = extract_answer(pattern_completion)
    t1 = time.time()
    if verbose:
        print('Pattern prompt:', pattern_prompt)
        print()
        print('Pattern response:', pattern_completion)
        print('Time (s):', t1 - t0)
        print()
    
    t0 = time.time()
    track_completion = sample_groq(system_prompt, track_prompt)
    track = extract_answer(track_completion)
    t1 = time.time()
    if verbose:
        print('Sample prompt:', track_prompt)
        print()
        print('Sample response:', track_completion)
        print('Time (s):', t1 - t0)
        print()
        
    return pattern, track

def query_hf(
    all_patterns: List[str],
    all_tracks: Dict[str, List[str]],
    time_signature,  # unused
    curr_layer_patterns_so_far: List[str],
    curr_layer_samples_so_far: List[str],
    prev_layers_so_far: List[str],
    emotion: str,
    verbose=True
):
    """
    Emotion -> pattern, sample using Hugging Face model
    Each query processes patterns and tracks based on emotion and context
    """
    def sample_hf(system_prompt, user_prompt):
        # Adjust prompt format for phi-2
        prompt = f"Instruct: {system_prompt}\n\n{user_prompt}\n\nAssistant:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the response after the prompt
        response = response.split("Assistant:")[-1].strip()
        return response

    extract_answer = lambda content: content.split('\n')[-1]
    
    system_prompt = get_system_prompt()
    pattern_prompt = get_pattern_prompt(emotion, all_patterns, curr_layer_patterns_so_far, prev_layers_so_far)
    track_prompt = get_track_prompt(emotion, all_tracks, curr_layer_samples_so_far, prev_layers_so_far)
    
    t0 = time.time()
    pattern_completion = sample_hf(system_prompt, pattern_prompt)
    pattern = extract_answer(pattern_completion)
    t1 = time.time()
    if verbose:
        print('Pattern prompt:', pattern_prompt)
        print()
        print('Pattern response:', pattern_completion)
        print('Time (s):', t1 - t0)
        print()
    
    t0 = time.time()
    track_completion = sample_hf(system_prompt, track_prompt)
    track = extract_answer(track_completion)
    t1 = time.time()
    if verbose:
        print('Sample prompt:', track_prompt)
        print()
        print('Sample response:', track_completion)
        print('Time (s):', t1 - t0)
        print()
        
    return pattern, track

if __name__ == "__main__":
    # Test all three model implementations
    patterns, synth_patterns = get_patterns()
    track_groups = get_track_groups()
    random_emotion = np.random.choice(all_emotion_str)
    print(f'Emotion: {random_emotion}')
    
    # Test each model
    for query_func in [query_openai, query_groq, query_hf]:
        print(f"\nTesting {query_func.__name__}:")
        try:
            pattern, track = query_func(
                all_patterns=patterns,
                all_tracks=track_groups,
                time_signature=None,
                curr_layer_patterns_so_far=[],
                curr_layer_samples_so_far=[],
                prev_layers_so_far=[],
                emotion=random_emotion,
                verbose=True
            )
            print(f"Results from {query_func.__name__}:")
            print(f"Pattern: {pattern}")
            print(f"Track: {track}")
        except Exception as e:
            print(f"Error running {query_func.__name__}: {str(e)}")