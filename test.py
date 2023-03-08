import time
import pathlib
from PIL import Image
from argparse import ArgumentParser

import torch
import faiss

from src.feature_extraction import MyResnet50, MyVGG16, RGBHistogram, LBP
from src.dataloader import get_transformation

ACCEPTED_IMAGE_EXTS = ['.jpg', '.png']

query_root = './dataset/groundtruth'
image_root = './dataset/paris'
feature_root = './dataset/feature'
evaluate_root = './dataset/evaluation'


def get_image_list(image_root):
    image_root = pathlib.Path(image_root)
    image_list = list()
    for image_path in image_root.iterdir():
        if image_path.exists():
            image_list.append(image_path)
    image_list = sorted(image_list, key = lambda x: x.name)
    return image_list

def main():

    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", required=True, type=str, default='VGG16')
    parser.add_argument("--device", required=False, type=str, default='cuda:0')
    parser.add_argument("--top_k", required=False, type=int, default=11)
    parser.add_argument("--test_image_path", required=False, type=str, default='./dataset/paris/paris_triomphe_001112.jpg')

    print('Start retrieving .......')
    start = time.time()

    args = parser.parse_args()
    device = torch.device(args.device)

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

    img_list = get_image_list(image_root)

    transform = get_transformation()

    # Preprocessing
    test_image_path = pathlib.Path(args.test_image_path)
    pil_image = Image.open(test_image_path)
    pil_image = pil_image.convert('RGB')
    image_tensor = transform(pil_image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    feat = extractor.extract_features(image_tensor)

    indexer = faiss.read_index(feature_root + '/' + args.feature_extractor + '.index.bin')
    
    _, indices = indexer.search(feat, k=args.top_k)
    print(indices)
    for index in indices[0]:
        print(img_list[index])

    end = time.time()
    print('Finish in ' + str(end - start) + ' seconds')

if __name__ == '__main__':
    main()