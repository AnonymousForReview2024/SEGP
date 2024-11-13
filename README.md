
# SEGP
The official PyTorch implementation of the paper SEGP.

# Overview
This repo contains the PyTorch implementation of SEGP, described in the paper SEGP.  

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


# Datasets
Download and extract [datasets](https://github.com/MediaBrain-SJTU/MVFA-AD?tab=readme-ov-file) into MVFA_data


# Train
&bull; baseline：python train_baseline.py --obj $target-object

&bull; SEGP：python train_maxNensemble.py --obj $target-object

# Test
&bull; python test.py --obj $target-object --save_path $path-model

# Experimental results
| method | OCT| HIS| Chest| Brain| Brain_p| Liver| Liver_p| RESC| RESC_p 
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- 
| baseline | 99.57| 83.58| 82.46| 90.94| 96.89| 86.94| 99.57| 95.86| 99.16 
| max5ensemble | 99.67| 83.84| 83.27| 91.74| 97.75| 87.13| 99.71| 96.29| 99.03 

# baseline models
&bull;OCT: link: https://pan.baidu.com/s/1Fj3EUwDIYb8q1gbajIVXlQ download code: kg3r

&bull;HIS: link: https://pan.baidu.com/s/1lH30m2qCbOHn2hdvwLznLw download code: cx86

&bull; Chest: link: https://pan.baidu.com/s/1W09KlCD2AMGHM4PY0LPETw download code: bd35

&bull; Brain: link: https://pan.baidu.com/s/1vcO7vXCKmiYQYuMSmvAkAw download code: kate

&bull; Liver: link: https://pan.baidu.com/s/1ctqk6s6TDxu3CenZEyn7sQ download code: mras

&bull; RESC: link: https://pan.baidu.com/s/1XNMd5ho9xSs12jkfYHkrxw download code: 424r

# max5ensemble models
&bull; OCT: link: https://pan.baidu.com/s/1FlrmIn3P0eopJMe0sg_pVA download code: h4r8

&bull; HIS: link: https://pan.baidu.com/s/1lCSIradrGNmovkIhMJVh6A download code: fgcs

&bull; Chest: link: https://pan.baidu.com/s/1bfJnJE0ipR5p9AE0QHq7IA download code: w2fq

&bull; Brain: link: https://pan.baidu.com/s/16f-J4LjIfXzQUSH1s533iA download code: 23v4

&bull; Liver: link: https://pan.baidu.com/s/1qk8gOViBWRHqeoNJWJORGQ download code: 53ij

&bull; RESC: link: https://pan.baidu.com/s/14j9m9_tmcTMpmejWUPALXg download code: cu27

# Acknowledgement
We borrow some codes from [OpenCLIP](https://github.com/mlfoundations/open_clip), and [MVFA](https://github.com/MediaBrain-SJTU/MVFA-AD?tab=readme-ov-file).

