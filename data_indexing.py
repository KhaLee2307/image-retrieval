from argparse import ArgumentParser

import faiss
import torch
from torch.utils.data import DataLoader, SequentialSampler

from util.feature_extraction import feature_extractor, extract_features
from util.indexing import get_faiss_indexer
from util.dataloader import MyDataLoader


def main():

    parser = ArgumentParser()
    parser.add_argument("--image_root", required=False, type=str, default='./data')
    parser.add_argument("--device", required=False, type=str, default='cuda:0')
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    device = torch.device(args.device)
    batch_size = args.batch_size

    model = feature_extractor()
    model = model.to(device)

    dataset = MyDataLoader(args.image_root)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,batch_size=batch_size,sampler=sampler)

    indexer = get_faiss_indexer()
    for indices, (images, image_paths) in enumerate(dataloader):
        images = images.to(device)
        features = extract_features(model, images)
        # print(features.shape)
        indexer.add(features)
    faiss.write_index(indexer, 'building.index.bin')

if __name__ == '__main__':
    main()