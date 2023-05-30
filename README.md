# TBIFormer
<!-- [Project Page](https://xiaogangpeng.github.io/TBIFormer/) |  -->
### [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Peng_Trajectory-Aware_Body_Interaction_Transformer_for_Multi-Person_Pose_Forecasting_CVPR_2023_paper.pdf) | [Video](https://) | [Project Page](http://xiaogangpeng.github.io/projects/tbiformer/page.html)
<br/
> Trajectory-Aware Body Interaction Transformer for Multi-Person Pose Forecasting  
> [Xiaogang Peng](https://xiaogangpeng.github.io/), Siyuan Mao, [Zizhao Wu*](http://zizhao.me)

## News
- [2022/4/20]: Initial code releases. 
- [2022/2/28]: Our paper is accepted by CVPR 2023. Thanks to my collaborators！

<!-- ## Demo  
Demos are a little bit large; please wait a moment to load them. If you cannot load them or feel them blurry, you can click the hyperlink of each demo for the full-resolution raw video. Welcome to the home page for more demos and detailed introductions. 

### [Occupancy prediction:](https://cloud.tsinghua.edu.cn/f/f7768f1f110c414297cc/)

<p align='center'>
<img src="./assets/demo1.gif" width="720px">
<img src="./assets/bar.jpg" width="720px">
</p>

### [Generated dense occupancy labels:](https://cloud.tsinghua.edu.cn/f/65d91a4c891f447da731/)
<p align='center'>
<img src="./assets/demo2.gif" width="720px">
</p> -->


## Introduction
Multi-person pose forecasting remains a challenging problem, especially in modeling fine-grained human body interaction in complex crowd scenarios. Existing methods typically represent the whole pose sequence as a temporal series, yet overlook interactive influences among people based on skeletal body parts. In this paper, we propose a novel Trajectory-Aware Body Interaction Transformer (TBIFormer) for multi-person pose forecasting via effectively modeling body part interactions. Specifically, we construct a Temporal Body Partition Module that transforms all the pose sequences into a Multi-Person Body-Part sequence to retain spatial and temporal information based on body semantics. Then, we devise a Social Body Interaction Self-Attention (SBI-MSA) module, utilizing the transformed sequence to learn body part dynamics for inter- and intra-individual interactions. Furthermore, different from prior Euclidean distance-based spatial encodings, we present a novel and efficient Trajectory-Aware Relative Position Encoding for SBI-MSA to offer discriminative spatial information and additional interactive clues. On both short- and long-term horizons, we empirically evaluate our framework on CMU-Mocap, MuPoTS-3D as well as synthesized datasets (6 ~ 10 persons), and demonstrate that our method greatly outperforms the state-of-the-art methods.

## Method 

Method Pipeline:

<p align='center'>
<img src="https://xiaogangpeng.github.io/images/TBIFormer_500x300.png" width="720px">
</p>

Prediction Results:
<p align='center'>
<img src="./assets/results.jpg" width="800px">
</p>
<p align='center'>
<img src="./assets/figure5.png" width="800px">
</p>


## Prepare Data
We mostly follow the preprocessing procedure of [MRT](https://github.com/jiashunwang/MRT) for mixing dataset. Due to the mixing has random operations, we have uploaded the mixed dataset and others for your convenience and fair comparision. The datasets can be downloade from [![Google Drive](https://img.shields.io/badge/Google-Drive-blue)](https://drive.google.com/file/d/1HM7pwrT_hxpqgjicAbhCKK45hTvWnC6F/view?usp=sharing). Please prepare your data like this:
```
project_folder/
├── checkpoints/
│   ├── ...
├── data/
│   ├── Mocap_UMPM
│   │   ├── train_3_75_mocap_umpm.npy
│   │   ├── test_3_75_mocap_umpm.npy
│   │   ├── test_3_75_mocap_umpm_shuffled.npy
│   ├── MuPoTs3D
│   │   ├── mupots_150_2persons.npy
│   │   ├── mupots_150_3persons.npy
│   ├── mix1_6persons.npy
│   ├── mix2_10persons.npy
├── models/
│   ├── ...
├── utils/
│   ├── ...
├── train.py
├── test.py
```
<!-- You can generate occupancy labels with or without semantics (via acitivating --with semantic). If your LiDAR is high-resolution, e.g. RS128, LiVOX and M1, you can skip Poisson reconstruction step and the generation processe will be very fast! You can change the point cloud range and voxel size in config.yaml. -->

## Training
```
python train.py
```
## Inference
```
python test.py
```

## Acknowledgement
Many thanks to the previous projects:
- [MRT](https://github.com/jiashunwang/MRT)
- [HRI](https://github.com/wei-mao-2019/HisRepItself)
- [MSR](https://github.com/Droliven/MSRGCN)

Related Projects:
- [iRPE](https://github.com/microsoft/Cream/tree/main/iRPE) (orignal code of piecewise index fuction)


## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@InProceedings{peng2023trajectory,
        author    = {Peng, Xiaogang and Mao, Siyuan and Wu, Zizhao},
        title     = {Trajectory-Aware Body Interaction Transformer for Multi-Person Pose Forecasting},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2023},
        pages     = {17121-17130}
}
```
