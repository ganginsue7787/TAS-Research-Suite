---
name: Bug Report
about: Report a bug or unexpected behavior
title: "[BUG] Brief description of the issue"
labels: bug
assignees: ''

---

## Description
Provide a clear and concise description of the bug you encountered.

## Steps to Reproduce
List the steps to reproduce the behavior:
1. 
2. 
3. 

## Expected Behavior
Describe what you expected to happen.

## Actual Behavior
Describe what actually happened instead.

## Environment
- **OS**: [e.g., Ubuntu 22.04, macOS 13, Windows 11]
- **Python Version**: [e.g., 3.10, 3.11, 3.12]
- **PyTorch Version**: [e.g., 2.0.0]
- **CUDA Version**: [e.g., 11.8, N/A for CPU]
- **Installation Method**: [e.g., pip install -e ".[dev,jupyter]"]

## Error Message / Stack Trace
If applicable, provide the complete error message and traceback:
```python
Traceback (most recent call last):
  File "...", line ..., in ...
    ...
Error: ...
```

## Code Example
Provide a minimal reproducible code example (MRE):
```python
import torch
import numpy as np
from source import tas_core

# Your code here
```

## Additional Context
Add any other relevant information, screenshots, or files that might help us diagnose the issue.

## Checklist
- [ ] I've searched existing issues to ensure this hasn't been reported
- [ ] I'm using the latest version of TAS-Research-Suite
- [ ] I've provided a minimal reproducible example
- [ ] I've included the complete error traceback
