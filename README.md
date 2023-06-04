# Content-Based Image Retrieval

<p align="center">
  <img src=demo.gif/>
</p>

## Introduction

This is my project built for the Content-based Image Retrieval problem. In this project, I use the algorithm of indexing and searching Faiss (Facebook). Simultaneously combine many feature extraction methods for comparison and evaluation (RGBHistogram, Local Binary Pattern, VGG16, ResNet50).

**Problem**

  - **Input**: A collection of information resources (image database), a query image
  - **Output**: A ranked list of images that are most similar to the query image, with the most similar image at the top of the list

<p align="center">
  <img src=diagram.png/>
</p>

I use the [faiss](https://github.com/facebookresearch/faiss.git) library created by facebook. The weights of the VGG16, Resnet50 networks are taken from the pre-trained model of [torchvision.models](https://pytorch.org/vision/stable/models.html).

## Prepare environment

1. python==3.8.16
2. Install pytorch-cuda==11.7 following [official instruction](https://pytorch.org/):

        conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
        
3. Install Facebook faiss:

        conda install -c conda-forge faiss-gpu
        
4. Install the necessary dependencies by running:

        pip install -r requirements.txt. 

## Prepare dataset

1. Put the downloaded [The Paris Dataset](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) in **./data/paris**

2. Put the downloaded [groundtruth](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) in **./data/groundtruth**

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

## Running the code

### Feature extraction (Indexing)

    python indexing.py --feature_extractor Resnet50
    
The Resnet50.index.bin file will be located at **Main-folder/dataset/feature**.

### Evaluation

Evaluation on query set

    python ranking.py --feature_extractor Resnet50
    
### Compute Mean Average Precision (MAP):

    python evaluate.py --feature_extractor Resnet50
    
### Run demo with streamlit interface:

    streamlit run demo.py
    
### Addition 

You can modify the config like feature_extractor (RGBHistogram, LBP, VGG16, Resnet50), batch_size, top_k, ...
