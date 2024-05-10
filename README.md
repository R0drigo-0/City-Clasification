# City Classification
This project aims to create a program that given a region of a map as an input, will output the city that it belongs to.

## Cities
- Barcelona
- Berlin
- Budapest
- London
- Madrid
- Paris
- Riga
- Stockholm
- Tallinn
- Zagreb

# How to use (V1, V2, V3)

# How to use (V4)

```commandline
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

git clone https://github.com/cvg/LightGlue.git && cd LightGlue

python -m pip install -e .
```

# Version 1
First version using Template Matching.

Limitations:
- Does not work with scale/rotation
- Works OK with classification

# Version 2
City Classifier
Second version using SIFT (Scale Invariant Feature Transform).
Improvements and limitations:
- Works with scale
- Works with rotation
- Works with positioning
- Is slower than V1

# Version 3
Work in progress

Experiment using other methods different from SIFT.

# Version 4
Work in progress

Improved version with Deep Learning ([LightGlue](https://github.com/cvg/LightGlue))