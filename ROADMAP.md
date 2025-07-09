# Project Roadmap: Mac Mini Connect

This document outlines the development plan, goals, and future vision for the Mac Mini Connect project.

## 1. Project Vision

To create a robust and observable distributed computing environment using two Mac Minis, capable of running and monitoring parallel processing tasks efficiently. The ultimate goal is to have a stable platform for experimenting with distributed machine learning models and other parallel workloads.

## 2. Current State (as of late 2024)

- **Network Foundation**: Successfully established reliable network connectivity between `mini-red` (master) and `mini-yellow` (worker).
- **Environment**: Python virtual environments (`venv`) are set up on both machines with necessary dependencies.
- **Deployment**: A `sync.sh` script is in place to synchronize the project codebase from the local development machine to both minis.
- **Core Test**: A basic PyTorch distributed sharding test (`test_sharding.py`) is confirmed to run successfully across the two-node cluster.
- **Troubleshooting**: Extensive network debugging has been performed and documented, resolving initial connectivity issues.

## 3. Short-Term Goals (Immediate Next Steps)

These are the tasks to be tackled next to build upon the current foundation.

-   [ ] **Run the Primary Application**: Execute the main `run_distributed.py` application and ensure it runs to completion without errors.
-   [ ] **Basic Benchmarking**: Measure the baseline performance of the `run_distributed.py` application (e.g., total execution time).
-   [ ] **Centralized Logging**: Implement a basic logging mechanism that collects the output from both nodes into a single, time-stamped log file for easier analysis.

## 4. Medium-Term Goals (Building Observability)

Once the core application is stable, the focus will shift to monitoring and visibility.

-   [ ] **Detailed Logging & Metrics**:
    -   Instrument the application code to log key events (e.g., task start/end, data transfer).
    -   Capture system utilization metrics (CPU, memory) from both nodes during a run.
-   [ ] **Network Traffic Visibility**:
    -   Implement tools or scripts to monitor the network traffic between the two minis during a distributed job.
    -   Analyze the volume and pattern of data being exchanged.
-   [ ] **Task Completion Visibility**:
    -   Develop a method to track and display the completion percentage of processing requests or batches in real-time or near-real-time.

## 5. Long-Term Goals (Creating a Full-Fledged Platform)

These goals aim to turn the project into a more permanent and user-friendly platform.

-   [ ] **Web Interface / Dashboard**:
    -   Design and build a simple web-based dashboard to visualize the data collected in the medium-term goals.
    -   Display real-time system utilization, network traffic, and job progress.
-   [ ] **Configuration Management**: Make the setup more flexible by moving hardcoded values (like IP addresses, hostnames) into a dedicated configuration file (`config.yaml` or similar).
-   [ ] **Automated Test Suite**: Create a suite of tests that can be run automatically to validate the environment and application stack after any changes. 