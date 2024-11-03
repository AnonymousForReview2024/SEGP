
# SEGP
The official PyTorch implementation of the paper XXXXXXXXXXXXXXXXXX.

# Overview
This repo contains the PyTorch implementation of SEGP, described in the paper XXXXXXXXXXXXXX.  

# datasets
Download and extract [datasets](https://github.com/MediaBrain-SJTU/MVFA-AD?tab=readme-ov-file) into MVFA_data


# Train
&bull; baseline：python train_baseline.py --obj $target-object

&bull; SEGP：python train_maxNensemble.py --obj $target-object

# Test
&bull; python test.py --obj $target-object --save_path $path-model

# Acknowledgement
We borrow some codes from [OpenCLIP](https://github.com/mlfoundations/open_clip), and [MVFA](https://github.com/MediaBrain-SJTU/MVFA-AD?tab=readme-ov-file).

