# Project Description
This repository contains the source code and instruction for Dense cotton boll counting. Detailed methodology and results can be found from our paper.

# Setup Instructions
Pip install all requirements in a Python>=3.8 environment with PyTorch>=1.8. 

# Usage
1. Clone this repository.
2. Clone the FlowFormer repository:<https://github.com/drinkingcoder/FlowFormer-Official> to the scr folder.
3. Create a Python virtual environment and install all requiremnets.
4. Download the trained detector or train your own data followed the RTDETR repository.
5. Run ./scr/boll_tracking.py to get the counting results.

# Data Information
The video dataset can be accessed on figshare:<https://figshare.com/s/cade84dfccf6ff6a67f9>. Please refresh the page if you cannot see the datasets.

# Acknowlegement
- [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official)
