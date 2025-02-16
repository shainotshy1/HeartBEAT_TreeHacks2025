# HeartBEAT_TreeHacks2025

## Setup
```bash
conda create -n heartbeat python=3.10
conda activate heartbeat
pip install -e .
```

## Data
```
mkdir data
cd data
wget https://www.hexawe.net/mess/200.Drum.Machines/drums.zip
unzip drums.zip
```

## Run

```bash
python heartbeat_scripts/generate.py
```

Currently this is set up so that we can test on the simulated heart rate data.

TODO (sort of in order):
* Tune the parameters so debugging in normal mode is less painful (haven't tested much, tbh have no idea if switching beats even works since I haven't run through this for more than a minute)
* Sounds weird cuz of some thread issue maybe? Or maybe my software is just bad.
* Implement synth (just uncomment stuff in `generate.py`)
* Add logging/visualization for original/filtered signal (so people have some idea of how we got to our emotion)
* Make it sound good (hard)
* Update arduino-related things