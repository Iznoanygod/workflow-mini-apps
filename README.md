# RADICAL Workflow Mini-Apps
Workflow Mini-Apps[^1] provides small, self-contained representations of scientific workflows (or mini-apps) for developing workflows.
Each mini-app is a simplified version of a complex scientific workflow, capturing its key tasks, data flow, and performance characteristics without the deployment challenges of the full application.
Workflow Mini-apps can be scaled and configured without application specific deployment challenges and constraints​.


Workflow Mini-app facilitate experimentation and helps understand workflow (distinct from application) performance.

There are 2 example Workflow Mini-apps:

- Neutron Diffraction Experiment (InverseProblem)

- AI Steered Simulations (DeepDriveMD)

### Installation
1). Install RADICAL tools. Please make sure to use conda env approach since we also need an env that has `cupy`/`h5py`/`mpi4py`

2). Install Darshan. Please make sure to modify the darshan code as explained so that it can be used to collect info. Also don't forget to install `darshan-util`

3). Set the environment, a sample script is shown below:

```shell
#/bin/bash

module load cray-hdf5/1.12.1.3
module load conda
conda activate <your RCT environment>

which python
python -V


export RADICAL_LOG_LVL=DEBUG
export RADICAL_PROFILE=TRUE
export RADICAL_SMT=1

export PATH=<path to darshan binary>:$PATH
```

Here `"<your RCT environment>"` is the conda env with RADICAL tools, and `"<path to darshan binary>"` is where Darshan is installed.

4). Go to the specific mini-app sub-dir, then do source `source_me.sh` 

5). Go to launch-scripts to run the experiment. Before starting, make sure the parameters have been set up

6). Analyze the results. Some useful tools can be found in Analyze/

This work has been supported by the DOE [RECUP](https://github.com/RECUP-DOE/workflow-miniapps) project.

[^1]: Ozgur O. Kilic, Tianle Wang, Matteo Turilli, Mikhail Titov, Andre Merzky, Line Pouchard, and Shantenu Jha (2024) "Workflow Mini-Apps: Portable, Scalable, Tunable & Faithful Representations of Scientific Workflows". https://doi.org/10.1109/CCGrid59990.2024.00059, https://arxiv.org/abs/2403.18073

