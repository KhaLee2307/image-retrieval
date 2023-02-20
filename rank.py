import os
import time
import pathlib
from PIL import Image
from argparse import ArgumentParser

import torch
import faiss

from src.feature_extraction import MyVGG16, MyResnet50, RGBHistogram, LBP
from src.dataloader import get_transformation

ACCEPTED_IMAGE_EXTS = ['.jpg', '.png']

query_root = './dataset/groundtruth'

def get_image_list(image_root):
    image_root = pathlib.Path(image_root)
    image_list = list()
    for image_path in os.listdir(image_root):
        image_list.append(image_path[:-4])
    image_list = sorted(image_list, key = lambda x: x)
    return image_list

def main():
    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", required=True, type=str, default='VGG16')
    parser.add_argument("--image_root", required=False, type=str, default = './dataset/paris')
    parser.add_argument("--device", required=False, type=str, default='cuda:0')
    parser.add_argument("--top_k", required=False, type=int, default=11)
    parser.add_argument("--crop", required=False, type=bool, default=False)


    print('Ranking .......')
    start = time.time()

    args = parser.parse_args()
    device = torch.device(args.device)

    if (args.feature_extractor == 'VGG16'):
        extractor = MyVGG16(device)
    elif (args.feature_extractor == 'Resnet50'):
        extractor = MyResnet50(device)
    elif (args.feature_extractor == 'RGBHistogram'):
        extractor = RGBHistogram(device)
    elif (args.feature_extractor == 'LBP'):
        extractor = LBP(device)
    else:
        print("No matching model found")
        return

    img_list = get_image_list(args.image_root)

    transform = get_transformation()

    for path_file in os.listdir(query_root):
        if (path_file[-9:-4] == 'query'):
            rank_list = []

            with open(query_root + '/' + path_file, "r") as file:
                img_query, left, top, right, bottom = file.read().split()

            test_image_path = pathlib.Path('./dataset/paris/' + img_query + '.jpg')
            pil_image = Image.open(test_image_path)
            pil_image = pil_image.convert('RGB')

            path_crop = 'original'
            if (args.crop):
                pil_image=pil_image.crop((float(left), float(top), float(right), float(bottom)))
                path_crop = 'crop'

            image_tensor = transform(pil_image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            feat = extractor.extract_features(image_tensor)

            indexer = faiss.read_index('./dataset/feature/' + args.feature_extractor + '.index.bin')

            _, indices = indexer.search(feat, k=args.top_k)  

            for index in indices[0]:
                rank_list.append(str(img_list[index]))

            with open('./dataset/evaluation/' + path_crop + '/' + args.feature_extractor + '/' + path_file[:-10] + '.txt', "w") as file:
                file.write("\n".join(rank_list))

    end = time.time()
    print('Finish in ' + str(end - start) + ' seconds')

if __name__ == '__main__':
    main()