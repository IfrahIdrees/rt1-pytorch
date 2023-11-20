#!/bin/bash

# https://github.com/pytorch/pytorch/pull/114083

# Find the PyTorch installation path using Python
pytorch_path=$(python -c "import os, torch; print(os.path.dirname(torch.__file__))")

# Construct the path to functional.py
functional_py="${pytorch_path}/nn/functional.py"

# Check if functional.py exists
if [ -f "$functional_py" ]; then
    # Apply the patch
    sed -i 's/.masked_fill_(mask,/.masked_fill_(~mask,/' "$functional_py"
    echo "Patch applied successfully to $functional_py"
else
    echo "functional.py not found in the PyTorch installation."
fi
