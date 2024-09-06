# Recalling Unknowns without Losing Precision: An Effective Solution to Large Model-Guided Open World Object Detection (TIP 2024)

<!-- [`Paper`](https://arxiv.org/abs/2212.01424)  -->

#### Yulin He, Wei Chen, Siqi Wang, Tianrui Liu, and Meng Wan


# Abstract

Open World Object Detection (OWOD) aims to adapt object detection to an open-world environment, so as to detect unknown objects and learn knowledge incrementally.  Existing OWOD methods typically leverage training sets with a relatively small number of known objects. Due to the absence of generic object knowledge, they fail to comprehensively perceive objects beyond the scope of training sets. Recent advancements in large vision models (LVMs), trained on extensive large-scale data, offer a promising opportunity to harness rich generic knowledge for the fundamental advancement of OWOD.  Motivated by Segment Anything Model (SAM), a prominent LVM lauded for its exceptional ability to segment generic objects, we first demonstrate the possibility to employ SAM for OWOD and establish the very first SAM-Guided OWOD baseline solution. Subsequently, we identify and address two fundamental challenges in SAM-Guided OWOD and propose a pioneering **S**AM-**G**uided **R**obust **O**pen-world **D**etector (SGROD) method, which can significantly improve the recall of unknown objects without losing the precision on known objects. Specifically, the two challenges in SAM-Guided OWOD include:
(1) Noisy labels caused by the class-agnostic nature of SAM;
(2) Precision degradation on known objects when more unknown objects are recalled.
For the first problem, we propose a dynamic label assignment (DLA) method that adaptively selects confident labels from SAM during training, evidently reducing the noise impact.
For the second problem, we introduce cross-layer learning (CLL) and SAM-based negative sampling (SNS), which enable SGROD to avoid precision loss by learning robust decision boundaries of objectness and classification.
Experiments on public datasets show that SGROD not only improves the recall of unknown objects by a large margin ($\sim 20$\%), but also preserves highly-competitive precision on known objects.

# Overview

- To the best of our knowledge, we are the first to propose exploiting the rich generic knowledge of large visual models (LVMs) to enhance OWOD. We demonstrate the feasibility of employing SAM for OWOD and establish the very first SAM-Guided OWOD baseline method.
- We identify and address three vital challenges in SAM-Guided OWOD, \textit{i.e.}, learning noisy labels from SAM by a dynamic label assignment (DLA) module, mitigating the optimization conflict between objectness and classification learning by a cross-layer learning (CLL) strategy, and preventing uncontrolled expansion of objectness semantics by a SAM-based negative sampling (SNS) module.
- Our proposed SGROD method significantly improves the recall of unknown objects while achieving robust performance on known object detection, which proves the feasibility and promise of leveraging LVMs to advance OWOD for handling open-world environments.

# Installation

### Requirements

```bash
conda create --name sgrod python==3.10.4
conda activate sgrod
pip install -r requirements.txt
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Backbone features

Download the self-supervised backbone (DINO) from [here](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth) and add in `models` folder.

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```




## Data Structure

```
SGROD/
└── data/
    └── OWOD/
        ├── JPEGImages
        ├── Annotations
        └── ImageSets
            ├── OWDETR
            ├── TOWOD
            └── VOC2007
```

### Dataset Preparation

The splits are present inside `data/OWOD/ImageSets/` folder.
1. Download the COCO Images and Annotations from [coco dataset](https://cocodataset.org/#download) into the `data/` directory.
2. Unzip train2017 and val2017 folder. The current directory structure should look like:
```
SGROD/
└── data/
    └── coco/
        ├── annotations/
        ├── train2017/
        └── val2017/
```
4. Move all images from `train2017/` and `val2017/` to `JPEGImages` folder.
5. Use the code `./datasets/coco2voc.py` for converting json annotations to xml files.
6. Download the PASCAL VOC 2007 & 2012 Images and Annotations from [pascal dataset](http://host.robots.ox.ac.uk/pascal/VOC/) into the `data/` directory.
7. untar the trainval 2007 and 2012 and test 2007 folders.
8. Move all the images to `JPEGImages` folder and annotations to `Annotations` folder. 
9. Download the pseudo labels of segment anything model (SAM) [SAM](https://github.com/facebookresearch/segment-anything) from [Annotations_segment](https://drive.google.com/file/d/1yT8nmarUmdcLDMB5IB-s1o6isdUk_Jsh/) into the `data/OWOD` directory.
You can also generate the pseudo labels by move `segment-anything/generate_proposal.py` to [SAM](https://github.com/facebookresearch/segment-anything) project and run it.

# Training

#### Training on single node

To train SGROD on a single node with 4 GPUS, run
```bash
bash ./run.sh
```
**note: you may need to give permissions to the .sh files under the 'configs' and 'tools' directories by running `chmod +x *.sh` in each directory.

By editing the run.sh file, you can decide to run each one of the configurations defined in ``\configs``:

1. EVAL_M_OWOD_BENCHMARK.sh - evaluation of tasks 1-4 on the MOWOD Benchmark.
2. EVAL_S_OWOD_BENCHMARK.sh - evaluation of tasks 1-4 on the SOWOD Benchmark. 
3. M_OWOD_BENCHMARK.sh - training for tasks 1-4 on the MOWOD Benchmark.
5. S_OWOD_BENCHMARK.sh - training for tasks 1-4 on the SOWOD Benchmark.




# Evaluation & Result Reproduction

For reproducing any of the aforementioned results, please download our [pretrain weights](https://drive.google.com/file/d/177wiKq19uxn9g42C9Z06Q7-n00uhuaca) and place them in the 
'checkpoints' directory. Run the `run_eval.sh` file to utilize multiple GPUs.

**note: you may need to give permissions to the .sh files under the 'configs' and 'tools' directories by running `chmod +x *.sh` in each directory.


```
SGROD/
└── checkpoints/
    ├── MOWODB/
    |   └── t1 checkpoint0040.pth
        └── t2_ft checkpoint0110.pth
        └── t3_ft checkpoint0180.pth
        └── t4_ft checkpoint0260.pth
    └── SOWODB/
        └── t1 checkpoint0040.pth
        └── t2_ft checkpoint0120.pth
        └── t3_ft checkpoint0200.pth
        └── t4_ft checkpoint0300.pth
```


**Note:**
For more training and evaluation details please check the [PROB](https://github.com/orrzohar/PROB) reposistory.




<!-- # Citation

If you use PROB, please consider citing:

```
@misc{zohar2022prob,
  author = {Zohar, Orr and Wang, Kuan-Chieh and Yeung, Serena},
  title = {PROB: Probabilistic Objectness for Open World Object Detection},
  publisher = {arXiv},
  year = {2022}
}
``` -->

# Contact

Should you have any question, please contact :e-mail: heyulin@nudt.edu.cn

**Acknowledgments:**

SGROD builds on previous works' code base such as [OW-DETR](https://github.com/akshitac8/OW-DETR), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [PROB](https://github.com/orrzohar/PROB), [SAM](https://github.com/facebookresearch/segment-anything.git), and [OWOD](https://github.com/JosephKJ/OWOD). If you found SGROD useful please consider citing these works as well.

