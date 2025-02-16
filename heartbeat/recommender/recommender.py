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

from heartbeat.heartbeat_sensor.emotion import all_emotion_str

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
    recent_patterns = curr_layer_patterns_so_far#[-5:]
    #patterns = [*all_patterns, *recent_patterns]
    patterns_concat_str = ', '.join(all_patterns)
    return f"Given that the emotion is {emotion}, and that we used the following recent patterns: [{recent_patterns}], and that the following progress towards track layers has already been done: [{prev_layers_so_far}], guess the best-matching drum pattern out of the following list. {get_expected_format_prompt()} \n{patterns_concat_str}"

def get_track_prompt(
    emotion: str, 
    all_tracks: List[str],
    curr_layer_samples_so_far: List[str],
    prev_layers_so_far: List[str]
):
    all_tracks = sample_list(all_tracks)
    recent_tracks = curr_layer_samples_so_far#[-5:]
    #tracks = [*all_tracks, *recent_tracks]
    tracks_concat_str = '\n'.join(all_tracks)
    return f"Given that the emotion is {emotion}, and that we used the following recent tracks: [{recent_tracks}], and that the following progress towards track layers has already been done: [{prev_layers_so_far}], guess the best-matching drum sample out of the following list. {get_expected_format_prompt()} \n{tracks_concat_str}"


def query_openai_old(emotion: str, 
                 time_signature,
                 curr_layer_patterns_so_far: List[str],
                 curr_layer_samples_so_far: List[str],
                 prev_layers_so_far: List[str],
                 verbose=False):
    """
    Emotion -> pattern, sample
    Each query takes ~1 second (depends on query length ofc)
    """
    raise NotImplementedError("Only groq is done")
    
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    get_content = lambda completion: completion.choices[0].message.content
    extract_answer = lambda content: content.split('\n')[-1]
    
    system_prompt = {"role": "system", "content": get_system_prompt()}
    pattern_prompt = {"role": "user", "content": get_pattern_prompt(emotion, patterns)}
    # synth_prompt = {"role": "user", "content": f"Given that the emotion is {emotion}, guess the best-matching synth pattern out of the following list. \n{', '.join(synth_patterns)}"}
    track_group_prompt = {"role": "user", "content": get_track_group_prompt(emotion, track_groups)}
    
    t0 = time.time()
    pattern_completion = get_content(openai_client.chat.completions.create(model="gpt-4o-mini", store=True, messages=[system_prompt, pattern_prompt], temperature=temperature))
    pattern = extract_answer(pattern_completion)
    t1 = time.time()
    if verbose:
        print('Drum pattern prompt:', pattern_prompt['content'])
        print()
        print('Drum pattern response:', pattern_completion)
        print('Time (s):', t1 - t0)
        print()
    
    t0 = time.time()
    track_group_completion = get_content(openai_client.chat.completions.create(model="gpt-4o-mini", store=True, messages=[system_prompt, track_group_prompt], temperature=temperature))
    track_group = extract_answer(track_group_completion)
    t1 = time.time()
    if verbose:
        print('Drum sample class prompt:', track_group_prompt['content'])
        print()
        print('Drum sample class response:', track_group_completion)
        print('Time (s):', t1 - t0)
        print()
    
    t0 = time.time()
    sample_filenames = get_samples_filenames(samples_data, track_group)
    sample_filename_prompt = {"role": "user", "content": get_sample_filename_prompt(emotion, track_group, sample_filenames)}
    sample_filename_completion = get_content(openai_client.chat.completions.create(model="gpt-4o-mini", store=True, messages=[system_prompt, sample_filename_prompt], temperature=temperature))
    sample_filename = extract_answer(sample_filename_completion)
    t1 = time.time()
    if verbose:
        print('Drum sample filename prompt:', sample_filename_prompt['content'])
        print()
        print('Drum sample filename response:', sample_filename_completion)
        print('Time (s):', t1 - t0)
        print()
    
    return {
        'pattern': pattern,
        'track_group': track_group,
        'sample_filename': sample_filename
    }
    

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
    if groq_api_key is None:
        raise ValueError("GROQ_API_KEY environment variable not set")
        
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
    

if __name__ == "__main__":
    raise NotImplementedError
    # patterns, synth_patterns = get_patterns()
    # track_groups = get_track_groups()
    # samples_data = get_samples_data()
    # random_emotion = np.random.choice(all_emotion_str)
    # print(f'Emotion: {random_emotion}')
    # # print(query_openai(random_emotion, patterns, synth_patterns, track_groups, samples_data, verbose=True))
    # print(query_groq(random_emotion, patterns, synth_patterns, track_groups, samples_data, verbose=True))