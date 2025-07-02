# Distributed Mac Mini ML Testing

This project tests distributed machine learning inference across multiple Mac Minis using PyTorch's distributed capabilities.

## Current Setup Progress

### Completed
- [x] Basic repository structure
- [x] SSH key setup from Ubuntu VM to both Macs
- [x] SSH config for easy access (`ssh mini-red` and `ssh mini-yellow`)
- [x] Homebrew installation on mini-red
- [x] Python 3.10 installation on mini-red

### Next Steps
- [ ] Install Homebrew and Python 3.10 on mini-yellow
- [ ] Set up project structure on both Macs:
  ```bash
  mkdir -p ~/projects
  cd ~/projects
  git clone https://github.com/chaceraiter/mac-mini-connect.git
  cd mac-mini-connect
  python3.10 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- [ ] Implement distributed ML testing framework
- [ ] Set up model sharding/parallelism

## Setup Instructions (Run on each Mac Mini)

1. Install Python 3.10+ if not already installed:
```bash
# Check if Python is installed
python3 --version

# If not installed, use brew
brew install python@3.10
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python3.10 -m venv venv

# Activate it
source venv/bin/activate
```

3. Install dependencies:
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

## Project Structure

```
.
├── README.md
├── requirements.txt
└── src/
    ├── node1/        # Primary node code
    └── node2/        # Secondary node code
```

## Network Configuration

Both Mac Minis are on the same local network:
- mini-red: 192.168.2.171
- mini-yellow: 192.168.2.224

SSH access is configured through the Ubuntu VM control node.

## Running the Tests

(Instructions to be added as we develop the testing scripts) 