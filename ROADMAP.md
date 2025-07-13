# Roadmap

## Current Status & Next Actions

**Context:** The primary goal is to run a distributed PyTorch application across two Mac Minis, `mini-red` (master) and `mini-yellow` (worker). After resolving initial connectivity issues, we've been blocked by a persistent `ModuleNotFoundError: No module named 'common'` when running the test script (`src/tests/test_sharding.py`).

**Exhaustive Debugging Summary:**
1.  **Package Structure:** Confirmed the project is a proper Python package with `__init__.py` files.
2.  **Execution Method:** Switched to the standard `python -m src.tests.test_sharding` invocation.
3.  **Environment Variable:** Explicitly set `PYTHONPATH` in various ways (`$PWD`, absolute paths) to point to the project root.
4.  **Python Interpreter:** Identified that the minis were using the system's problematic `/usr/bin/python3`. We rebuilt the virtual environments from scratch using a stable Homebrew Python (`/opt/homebrew/bin/python3.10`) via an Ansible playbook.

**Diagnosis:** Despite these corrective actions, the `ModuleNotFoundError` persists. We've concluded that the remote shell environment on the Mac Minis is fundamentally unreliable and is likely ignoring, unsetting, or misinterpreting the `PYTHONPATH` variable.

**Immediate Plan:**
*   **Implement a programmatic workaround:** Modify the Python test script (`src/tests/test_sharding.py`) to programmatically add the project's root directory to `sys.path`. This will make the script self-sufficient, bypassing the problematic remote shell environment and ensuring that local modules like `common` can be found reliably.

***

## Phase 1: Foundational Setup (Complete)

To create a robust and observable distributed computing environment using two Mac Minis, capable of running and monitoring parallel processing tasks efficiently. The ultimate goal is to have a stable platform for experimenting with distributed machine learning models and other parallel workloads.

## 2. Current State (as of late 2024)

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