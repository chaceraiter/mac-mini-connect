# Distributed Mac Mini ML Testing

This project tests distributed machine learning inference across multiple Mac Minis using PyTorch's distributed capabilities.

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
python3 -m venv venv

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

Both Mac Minis should be on the same local network and able to communicate via SSH.

## Running the Tests

(Instructions to be added as we develop the testing scripts) 