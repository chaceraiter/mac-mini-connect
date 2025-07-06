# Distributed Mac Mini ML Testing

This project tests distributed machine learning inference across multiple Mac Minis using PyTorch's distributed capabilities.

## Project Overview

The project consists of two main components:

1. **Basic Network Testing** (`scripts/test_connection.py`):
   - Tests raw socket connectivity between Mac Minis
   - Used for debugging network issues
   - Currently experiencing "No route to host" errors that appear related to TCP connection behavior
   - Interesting behavior where connections work when tcpdump is running

2. **Distributed ML System** (`src/`):
   - Uses PyTorch's distributed computing with GLOO backend (NCCL not available on macOS)
   - Implements model sharding across machines
   - Pipeline parallelism for large language model inference
   - Currently testing with OPT-350M model

## Project Structure
```
.
├── README.md
├── requirements.txt        # Python dependencies
├── scripts/               # Testing and utility scripts
│   ├── test_connection.py # Basic network connectivity test
│   ├── test_network.sh    # Network test launcher
│   ├── run_test.sh       # ML test launcher
│   └── sync.sh           # Code sync utility
├── ansible/               # Infrastructure automation
│   ├── ansible.cfg       # Ansible configuration
│   ├── inventory/        # Node definitions
│   └── playbooks/        # Automation tasks
└── src/                  # Source code
    ├── common/           # Shared utilities
    │   ├── __init__.py
    │   ├── config.py     # Node and model configurations
    │   └── distributed.py # Distributed computing utilities
    ├── model_sharding.py # Pipeline parallelism implementation
    ├── test_sharding.py  # Test harness for distributed setup
    └── run_distributed.py # Main distributed script
```

## Current Setup Progress

### Completed
- [x] Basic repository structure
- [x] SSH key setup from control machine to both Macs
- [x] SSH config for easy access (`ssh mini-red` and `ssh mini-yellow`)
- [x] Homebrew installation on both Macs
- [x] Python 3.10 installation on both Macs
- [x] Project structure setup on both Macs
- [x] Ansible playbooks for setup and updates
- [x] Basic distributed computing utilities
- [x] Node configuration and communication setup
- [x] Initial model sharding implementation
- [x] Network configuration fixes:
  - [x] Switched from NCCL to gloo backend for macOS compatibility
  - [x] Fixed distributed binding issues by using "0.0.0.0" for master node
  - [x] Configured proper master node IP advertisement

### Current Issues
- [ ] Basic socket connectivity test fails with "No route to host"
  - Works when tcpdump is running
  - Affects both Python socket test and nc command
  - Network configuration appears correct (ping works, DNS resolves)
- [ ] Need to verify distributed ML system functionality
  - GLOO backend configuration
  - Model sharding implementation
  - Inter-node communication

### Next Steps
- [ ] Resolve network connectivity issues
- [ ] Create development launcher script for quick testing
- [ ] Create ML task playbooks for production runs
- [ ] Implement distributed ML testing framework:
  - [ ] Model loading and sharding
  - [ ] Data pipeline setup
  - [ ] Performance metrics collection
- [ ] Add monitoring and logging
- [ ] Set up error handling and recovery
- [ ] Add CI/CD pipeline

## Network Configuration

Both Mac Minis are on the same local network:
- mini-red: 192.168.2.171 (Primary node, rank 0)
- mini-yellow: 192.168.2.224 (Secondary node, rank 1)

### Network Testing
The project includes basic network connectivity tests:
```bash
# On mini-red (server)
python scripts/test_connection.py server

# On mini-yellow (client)
python scripts/test_connection.py mini-red.lan
```

### Distributed ML Testing
For the actual distributed ML system:
```bash
# Start the test on both machines
./scripts/run_test.sh
```

## Setup Instructions

### Initial Setup
```bash
# Clone repository
git clone https://github.com/chaceraiter/mac-mini-connect.git
cd mac-mini-connect

# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### SSH Configuration
Ensure SSH access is configured:
```bash
# Generate SSH key for automation
ssh-keygen -t ed25519 -f ~/.ssh/mac_mini_key -C "mac-mini-connect"

# Copy to nodes
ssh-copy-id -i ~/.ssh/mac_mini_key.pub mini-red@192.168.2.171
ssh-copy-id -i ~/.ssh/mac_mini_key.pub mini-yellow@192.168.2.224
```

## Development vs Production

### Development Workflow
- Use test scripts directly for:
  - Quick iterations
  - Real-time output monitoring
  - Interactive debugging
  - Local testing

### Production Workflow
- Use Ansible playbooks for:
  - Deployment
  - Production runs
  - System monitoring
  - Log collection

## Monitoring and Logs

Currently implemented:
- Debug logging in test scripts
- Network diagnostics output
- Process status monitoring

Coming soon:
- Performance metrics
- Node status
- Error logging
- Resource utilization 