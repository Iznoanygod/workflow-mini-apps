# RADICAL Workflow Mini-App
Mini-apps development repo

WF Mini-apps can be scaled and configured without application specific deployment challenges and constraints​

WF Mini-app facilitate experimentation, understand workflow (distinct from application) performance​

We develop 2 WF mini-apps:
 
 - WF1: Scalable Adaptive Learning Learning (ExaLearn)​
 
 - WF2: DeepDriveMD​


ExaLearn mini-app implementation:

- Task1: Data Generation
    - Does a random matrix multiplication and prints the result to a file
- Task2: ML
    - Reads matrices created by data generation and does a matrix multiplication
    - For the next step will run a simplified ML algorithm that has 1-2 layers.
-enTK implementations:
    - Serial: DG1 -> ML1 -> DG2 -> ML2 -> ... -> DGn -> MLn
    - Parallel: DG1-> (ML1, DG2) -> (ML2, DG3) -> ... -> (MLn-1, DGn) -> MLn
- Requirements:
    - \#Tasks: 8 ( 4 of each)
    - Currently static, will be adaptive
    - python executables
    - \# nodes: 128 (real app) Currently 1 for mini-app
    - \# cores: 64 per node
    - \# gpus: currently only CPU but will move to gpu soon (~128 per node)
    - Has input/output dependencies
    - Original ExaLearn Generates ~70GB of data during Data Generation and reads it during ML
    - Data generated and read in the same node. No multi-node data sharing
    - Systems: Currently Theta, will move to Polaris


