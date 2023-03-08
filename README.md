# Content-Based Image Retrieval

## Introduction

This is a project we built for the subject CS336 - Multimedia Information Retrieval (University of Information Technology - VNUHCM). In this project, we use the algorithm of indexing and searching Faiss (Facebook). Simultaneously combine many feature extraction methods for comparison and evaluation (RGBHistogram, Local Binary Pattern, VGG16, ResNet50).

**Problem**

  - **Input**: A collection of information resources (image database), a query image
  - **Output**: A ranked list of images that are most similar to the query image, with the most similar image at the top of the list

<p align="center">
  <img src=diagram.png/>
</p>

We use the [faiss](https://github.com/facebookresearch/faiss.git) library created by facebook. The weights of the VGG16, Resnet50 networks are taken from the pre-trained model of [torchvision.models](https://pytorch.org/vision/stable/models.html).

## Prepare environment

1. python==3.8.16
2. Install pytorch-cuda==11.7 following [official instruction](https://pytorch.org/):

        conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
        
3. Install Facebook faiss:

        conda install -c conda-forge faiss-gpu
        
4. Install the necessary dependencies by running:

        pip install -r requirements.txt. 

## Prepare dataset

Please organizing your dataset following this structure: 

```
Main-folder/
│
├── dataset/ 
│   ├── evaluation
|   |   ├── crop
|   |   |   ├── LBP
|   |   |   |   ├── defense_1.txt
|   |   |   |   ├── eiffel_1.txt
|   |   |   |   └── ...
|   |   |   ├── Resnet50
|   |   |   |   └── ...
|   |   |   ├── RGBHistogram
|   |   |   |   └── ...
|   |   |   └── VGG16
|   |   |       └── ...
|   |   └── original
|   |       └── ...
|   |
│   ├── feature
|   |   ├── LBP.index.bin
|   |   ├── Resnet50.index.bin
|   |   ├── RGBHistogram.index.bin
|   |   └── VGG16.index.bin
|   |   
|   ├── groundtruth
|   |   ├── defense_1_good.txt
|   |   ├── eiffel_4_query.txt
|   |   └── ...
|   |
|   └── paris
|       ├── paris_defense_000000.jpg
|       ├── paris_defense_000042.jpg
|       └── ...
|   
└── ...
```

1. Put the downloaded [FreiHAND](https://github.com/lmb-freiburg/freihand) dataset in **./data/**

Link: https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip

2. Put the downloaded [FreiHAND](https://github.com/lmb-freiburg/freihand) evaluation set in **./data/**

Link: https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip

## Running the code

### Training
In this project, we focus on training Stacked Hourglass Network. As for the hand detect module, we reuse the victordibia's pretrained_model (SSD) without further modified. Train the hourglass network:

    python 1.train.py --config-file "configs/train_FreiHAND_dataset.yaml"
    
The trained model weights (net_hm.pth) will be located at **Main-folder/**. Simply copy and paste the trained model into **./model/trained_models** before evaluate.

### Evaluation

Evaluate on FreiHAND dataset:

    python 2.evaluate_FreiHAND.py --config-file "configs/eval_FreiHAND_dataset.yaml"
    
The visualization results will be saved to **./output/**

### Real-time hand pose estimation

Prepare camera and clear angle, good light, less noisy space. Run the following command line:

    python 3.real_time_2D_hand_pose_estimation.py --config-file "configs/eval_webcam.yaml"
    
_Note: Our model only solves the one-handed recognition problem. If there are 2 or more hands, the model will randomly select one hand to predict. To predict multiple hands, please edit the file 3.real_time_2D_hand_pose_estimation.py (because of resource and time limitations, we don't do this part)._

### Addition

To fine-tune the hyperparameters (BATCH_SIZE, NUM_WORKERS, DATA_SIZE, ...), you can edit the .yaml files in the **./configs/** directory.

## Citation

Newell, Alejandro & Yang, Kaiyu & Deng, Jia. (2016). [Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/pdf/1603.06937.pdf). 9912. 483-499. 10.1007/978-3-319-46484-8_29.
