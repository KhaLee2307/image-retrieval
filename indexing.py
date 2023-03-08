import time
from argparse import ArgumentParser

import faiss
import torch
from torch.utils.data import DataLoader, SequentialSampler

from src.feature_extraction import MyResnet50, MyVGG16, RGBHistogram, LBP
from src.indexing import get_faiss_indexer
from src.dataloader import MyDataLoader

image_root = './dataset/paris'
feature_root = './dataset/feature'


def main():

    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", required=True, type=str, default='Resnet50')
    parser.add_argument("--device", required=False, type=str, default='cuda:0')
    parser.add_argument("--batch_size", required=False, type=int, default=64)

    print('Start indexing .......')
    start = time.time()

    args = parser.parse_args()
    device = torch.device(args.device)
    batch_size = args.batch_size

    # Load module feature extraction 
    if (args.feature_extractor == 'Resnet50'):
        extractor = MyResnet50(device)
    elif (args.feature_extractor == 'VGG16'):
        extractor = MyVGG16(device)
    elif (args.feature_extractor == 'RGBHistogram'):
        extractor = RGBHistogram(device)
    elif (args.feature_extractor == 'LBP'):
        extractor = LBP(device)
    else:
        print("No matching model found")
        return

    dataset = MyDataLoader(image_root)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,batch_size=batch_size,sampler=sampler)

    indexer = get_faiss_indexer(extractor.shape)

    for images, image_paths in dataloader:
        images = images.to(device)
        features = extractor.extract_features(images)
        # print(features.shape)
        indexer.add(features)
    
    # Save features
    faiss.write_index(indexer, feature_root + '/' + args.feature_extractor + '.index.bin')
    
    end = time.time()
    print('Finish in ' + str(end - start) + ' seconds')

if __name__ == '__main__':
    main()