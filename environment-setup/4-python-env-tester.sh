# Check which Python the shell will use
which python3
python3 -c "import sys; print('exe:', sys.executable); print('ver:', sys.version.split()[0])"

# Check pip for that Python and whether torch is installed
python3 -m pip --version
python3 -m pip show torch || echo "torch not found for this python"

# Install torch into that exact interpreter if missing
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision

# Create and use a virtualenv (recommended)
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision

