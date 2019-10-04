#### AI-based Air Traffic Conflict Resolution

Preliminary investigation by me, as part of a larger group project.

Currently working on getting BlueSky running. By design, BlueSky can take in time-series scenario files (`.SCN`), but we want it to be able to stream data back and forth to Python.

#### INSTALLING (General)
Installing dependencies,
- On **MacOS** with Python 3.6x, use `pip install -r requirements.txt`
- On **MacOS** with Python 3.7x, check `README_VENV.md` for instructions.
- On **Windows/Linux** you can install the Python(x,y) bundle which has all the dependencies. See resources below.

Running BlueSky,
- Run `python check.py` to verify BlueSky installation.
- Run `python BlueSky.py` to launch the gui.

#### RESOURCES
**Installation and Guides** - <br>
BlueSky Wiki : https://github.com/TUDelft-CNS-ATM/bluesky/wiki
BlueSky Installation : http://homepage.tudelft.nl/7p97s/BlueSky/download.html
<br><br>
**Other** - <br>
BlueSky can be integrated with BADA, which models the performance of different aircraft types. Can be used to set state-action constraints and costs.
https://simulations.eurocontrol.int/solutions/bada-aircraft-performance-model/