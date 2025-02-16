from enum import Enum
import threading
import pygame

class Note(Enum):
    THIRTYSECOND = 1
    SIXTEENTH = 2
    EIGHTH = 3
    QUARTER = 4
    HALF = 5
    WHOLE = 6

note_to_count = {
    Note.THIRTYSECOND : 32, 
    Note.SIXTEENTH : 16,
    Note.EIGHTH : 8,
    Note.QUARTER : 4,
    Note.HALF : 2,
    Note.WHOLE : 1
}

class TimeSignature():
    # a / b time => 6/8 is a = 6, b = 8
    def __init__(self, a, b):
        assert b in note_to_count.values()
        assert type(a) is int
        self.a = a
        self.b = b

class BPM_Child():
    def step(self):
        raise NotImplemented
    
    def get_base_time(self):
        raise NotImplemented

class Sample():
    def __init__(self, length, label):
        assert type(length) is Note
        self.length = length
        self.label = label

    def play(self):
        return
        #print(f"Playing {self.label}")

    def empty_sample(length):
        return Sample(length, "<EMPTY>")

class WavSample(Sample):
    def __init__(self, length, label, wav_fn):
        super().__init__(length, label)
        self.wav_fn = wav_fn
        if not pygame.mixer.get_init():
            pygame.mixer.init(buffer=256)
        self.sound = pygame.mixer.Sound(self.wav_fn)

    def play(self):
        def helper():
            channel = pygame.mixer.find_channel()
            if channel:
                channel.play(self.sound)
        sound_thread = threading.Thread(target=helper)
        sound_thread.start()

class BPM_Manager():
    def __init__(self, bpm, metronome_sample = None, beat_note = Note.QUARTER, base_unit = Note.THIRTYSECOND):
        assert type(bpm) is int
        assert type(beat_note) is Note
        assert bpm > 0
        assert note_to_count[beat_note] <= note_to_count[base_unit]
        assert metronome_sample is None or isinstance(metronome_sample, Sample)
        self.bpm = bpm
        self.metronome_sample = metronome_sample
        self.beat_note = beat_note
        self.base_unit = base_unit
        self.children = []

    def add_child(self, child):
        assert isinstance(child, BPM_Child)
        assert child.get_base_time() == self.base_unit
        self.children.append(child)

    def remove_child(self, child):
        self.children.remove(child)

    def start(self):
        self.running = True
        self.tick()

    def stop(self):
        self.running = False

    def tick(self):
        sample_interval = note_to_count[self.base_unit] / note_to_count[self.beat_note]
        total_beats = self.bpm * sample_interval
        beat_interval = 60 / total_beats
        curr = 0
        def helper():
            nonlocal curr
            if not self.running:
                return
            if self.metronome_sample is not None and curr == 0:
                self.metronome_sample.play()
            curr = (curr + 1) % sample_interval
            for child in self.children:
                child.step()
            threading.Timer(beat_interval, helper).start()
        helper()

class SampleLibrary():
    def __init__(self, samples):
        self.samples = set()
        for s in samples:
            self.add_sample(s)

    def add_sample(self, sample):
        assert isinstance(sample, Sample)
        self.samples.add(sample)

class BeatBar(BPM_Child):
    # Example: 6/8 with base_time 1/16 => bar = [len = 6 * 16 / 8] = [12 sixteenth notes per bar]
    def __init__(self, time_signature, base_time):
        self.base_time = base_time
        self.time_signature = time_signature
        self.curr = 0
        n = self.get_total_time_units(base_time)
        self.bar = [Sample.empty_sample(base_time) for _ in range(n)]
    
    def set_count_sample(self, count, sample):
        assert 0 <= count < len(self.bar)
        assert isinstance(sample, Sample)
        self.bar[count] = sample

    def get_bar_arr(self):
        return self.bar

    def load_pattern(self, arr, sample):
        assert isinstance(sample, Sample)
        assert type(arr) == list
        assert len(arr) <= len(self.bar)
        assert len(self.bar) % len(arr) == 0
        interval = len(self.bar) // len(arr)
        for i, elem in enumerate(arr):
            if elem:
                self.set_count_sample(i * interval, sample)

    def step(self):
        self.bar[self.curr].play()
        self.curr = self.curr + 1
        if self.curr == len(self.bar): 
            self.curr = 0 # finished bar
            return True
        return False
    
    def get_base_time(self):
        return self.base_time

    # How many 'unit's fit in one bar
    def get_total_time_units(self, unit):
        a, b = self.time_signature.a, self.time_signature.b
        assert type(unit) is Note
        count = note_to_count[unit]
        assert count >= b
        return a * count // b

class BeatLayer(BPM_Child):
    def __init__(self, bars):
        assert type(bars) is list
        assert len(bars) > 0
        assert all(type(b) is BeatBar for b in bars)
        base_time = bars[0].base_time
        assert all(b.get_base_time() == base_time for b in bars)
        self.bars = bars
        self.curr = 0
        self.base_time = base_time

    def step(self):
        finished_bar = self.bars[self.curr].step()
        if finished_bar:
            self.curr = (self.curr + 1) % len(self.bars)

    def get_total_time_units(self, unit):
        return sum(b.get_total_time_units(unit) for b in self.bars)
    
    def get_base_time(self):
        return self.base_time
    
class Beat(BPM_Child):
    def __init__(self, layers):
        assert type(layers) is list
        assert len(layers) > 0
        assert all(type(l) is BeatLayer for l in layers)
        base_time = layers[0].base_time
        assert all(l.get_base_time() == base_time for l in layers)
        total_time = layers[0].get_total_time_units(base_time)
        assert all(l.get_total_time_units(base_time) == total_time for l in layers)
        self.layers = layers
        self.base_time = base_time

    def step(self):
        for layer in self.layers:
            layer.step()

    def get_base_time(self):
        return self.base_time

class BeatConstructor():
    def __init__(self, library):
        assert type(library) is SampleLibrary
        self.library = library

    def build_beat(self, layer_names, time_signature, num_bars):
        # FOR PRESTON TO IMPLEMENT: Use BeatBar.set_count_sample()
        layers = []
        return Beat(layers)
    