
# SEGP
The official PyTorch implementation of the paper XXXXXXXXXXXXXXXXXX.

# Overview
This repo contains the PyTorch implementation of SEGP, described in the paper XXXXXXXXXXXXXX.  

# Environment
&bull; python >= 3.8.5<br>
&bull; pytorch >= 1.10.0<br>
&bull; torchvision >= 0.11.1<br>
&bull; numpy >= 1.19.2<br>
&bull; scipy >= 1.5.2<br>
&bull; kornia >= 0.6.1<br>
&bull; pandas >= 1.1.3<br>
&bull; opencv-python >= 4.5.4<br>
&bull; pillow<br>
&bull; tqdm<br>
&bull; ftfy<br>
&bull; regex<br>
&bull; cuml<br>


# datasets
Download and extract [datasets](https://github.com/MediaBrain-SJTU/MVFA-AD?tab=readme-ov-file) into MVFA_data


# Train
&bull; baseline：python train_baseline.py --obj $target-object

&bull; SEGP：python train_maxNensemble.py --obj $target-object

# Test
&bull; python test.py --obj $target-object --save_path $path-model

# Acknowledgement
We borrow some codes from [OpenCLIP](https://github.com/mlfoundations/open_clip), and [MVFA](https://github.com/MediaBrain-SJTU/MVFA-AD?tab=readme-ov-file).

