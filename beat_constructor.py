from enum import Enum

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

class Sample():
    def __init__(self, sample, length, label):
        assert type(length) is Note
        assert label != "EMPTY", "EMPTY label is reserved, use another label"
        self.sample = sample
        self.length = length
        self.label = label

    def play(self):
        # Make play the audio and return after the length has completed
        if self.sample is not None:
            print(f"Playing {self.label}")

    def empty_sample(length):
        return Sample(None, length, "EMPTY")

class SampleLibrary():
    def __init__(self, samples):
        self.samples = set()
        for s in samples:
            self.add_sample(s)

    def add_sample(self, sample):
        assert type(sample) is Sample
        self.samples.add(sample)

class BeatBar():
    # Example: 6/8 with base_time 1/16 => bar = [len = 6 * 16 / 8] = [12 sixteenth notes per bar]
    def __init__(self, time_signature, base_time):
        n = self.get_total_time_units(base_time)
        self.bar = [Sample.empty_sample() for _ in range(n)]
        self.base_time = base_time
        self.time_signature = time_signature
    
    def set_count_sample(self, count, sample):
        assert 0 <= count < len(self.bar)
        assert type(sample) is Sample
        self.bar[count] = sample

    def get_bar_arr(self):
        return self.bar

    def play(self):
        for s in self.bar:
            s.play()

    # How many 'unit's fit in one bar
    def get_total_time_units(self, unit):
        a, b = self.time_signature.a, self.time_signature.b
        assert type(unit) is Note
        count = note_to_count[unit]
        assert count <= b
        return a * count // b

class BeatLayer():
    def __init__(self, bars):
        assert type(bars) is list
        for b in bars:
            assert type(b) is BeatBar
        self.bars = bars

    def play(self):
        for b in self.bars:
            b.play()

    def get_total_time_units(self, unit):
        return sum(b.get_total_time_units(unit) for b in self.bars)

class Beat():
    def __init__(self, layers):
        assert type(layers) is list
        assert len(layers) > 0
        for l in layers:
            assert(type(l) is BeatLayer)
        total_time = layers[0].get_total_time_units()
        assert all(l.get_total_time_units() == total_time for l in layers)
        self.layers = layers

    def play(self):
        for layer in self.layers:
            layer.play()

class BeatConstructor():
    def __init__(self, library):
        assert type(library) is SampleLibrary
        self.library = library

    def build_beat(self, layer_names, time_signature, num_bars):
        # FOR PRESTON TO IMPLEMENT: Use BeatBar.set_count_sample()
        layers = []
        return Beat(layers)
    