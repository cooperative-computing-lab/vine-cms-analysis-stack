# Vine CMS Example Stack

This is what works for CMS analysis with coffea on top of taskvine. No dependency of Notre Dame resources so it should work in other places.

## Installation

### Prerequisites

This project requires Python 3.13+. Dependencies are declared both as a conda `environment.yml` and as a [pixi](https://pixi.sh) workspace in `pyproject.toml`, so you can use whichever tool you already have.

### Option A: Using conda (recommended)

1. **Create the conda environment from the provided environment.yml file:**
   ```bash
   conda env create -f environment.yml -n vine-cms-example-stack
   ```

2. **Activate the environment:**
   ```bash
   conda activate vine-cms-example-stack
   ```

3. **Verify the installation:**
   ```bash
   python --version
   conda list | grep -E "(coffea|ndcctools)"  # Should show the installed packages
   ```

4. **Install vine\_reduce:**
   ```bash
   # Clone the repository
   git clone https://github.com/cooperative-computing-lab/vine_reduce.git
   cd vine_reduce

   # Activate the conda environment (if not already active)
   conda activate vine-cms-example-stack

   # Install the package
   pip install .
   ```

### Option B: Using pixi

1. **Install pixi**, if you don't already have it:
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. **Clone the repository and install the environment:**
   ```bash
   git clone https://github.com/cooperative-computing-lab/vine_reduce.git
   cd vine_reduce

   # Creates the environment defined in pyproject.toml's [tool.pixi] sections
   # and installs vine_reduce itself in editable mode.
   pixi install
   ```

3. **Run commands inside the environment**, either one at a time:
   ```bash
   pixi run python --version
   pixi run python examples/quick_start/quick.py
   ```
   or by dropping into an activated shell for the rest of the session:
   ```bash
   pixi shell
   python --version
   ```

4. **Verify the installation:**
   ```bash
   pixi run python --version
   pixi list | grep -E "(coffea|ndcctools)"  # Should show the installed packages
   ```


## Quick Start

Minimal toy example to get started. It's also available as a standalone script at
[examples/quick_start/quick.py](https://github.com/cooperative-computing-lab/vine_reduce/blob/main/examples/quick_start/quick.py),
which you can run directly with `pixi run python examples/quick_start/quick.py`.

```python
from vine_reduce import VineReduce
import ndcctools.taskvine as vine
import getpass

# Simple data: process two datasets
data = {
    "datasets": {
        "numbers": {"values": [1, 2, 3, 4, 5]},
        "more_numbers": {"values": [10, 20, 30]}
    }
}

# Define functions
def preprocess(dataset_info, **kwargs):
    for val in dataset_info["values"]:
        yield (val, 1)

def postprocess(val, **kwargs):
    return val  # Just return the value

def processor(x):
    return x * 2  # Double each number

def reducer(a, b):
    return a + b  # Sum the results

# Run
mgr = vine.Manager(port=[9123, 9129], name=f"{getpass.getuser()}-quick-start-vr")
print(f"Manager started on port {mgr.port}")
vr = VineReduce(mgr,
                           data=data,
                           source_preprocess=preprocess, 
                           source_postprocess=postprocess,
                           processors=processor, 
                           accumulator=reducer)

# Use local workers, condor, slurm, or sge for scale
workers = vine.Factory("local", manager=mgr)
workers.max_workers = 2
workers.min_workers = 0
workers.cores = 4
workers.memory = 2000
workers.disk = 8000
with workers:
    result = vr.compute()

print(f"Result: {result}")
# Expected: {'processor': {'numbers': 30, 'more_numbers': 120}}
# i.e. (1+2+3+4+5)*2 = 30 and (10+20+30)*2 = 120, one entry per dataset
```

## Physics Example ttBar

activate env, install from github source using pip install .https://github.com/TopEFT/topcoffea.git and https://github.com/TopEFT/ttbarEFT.git 

## Usage

- General use example: [examples/quick_start/quick.py](https://github.com/cooperative-computing-lab/vine_reduce/blob/main/examples/quick_start/quick.py)
- Using Coffea Processor Classes Directly: [examples/ttBar/run_processor_with_vr.py](https://github.com/cooperative-computing-lab/vine_reduce/blob/main/examples/ttBar/run_processor_with_vr.py)
- Coffea use in analysis: [examples/cortado/vr_cortado.py](https://github.com/cooperative-computing-lab/vine_reduce/blob/main/examples/cortado/vr_cortado.py)


## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
