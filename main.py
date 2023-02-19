import pathlib
from argparse import ArgumentParser

from PIL import Image

import torch
import faiss

from src.feature_extraction import MyVGG16, RGBHistogram
from src.dataloader import get_transformation

ACCEPTED_IMAGE_EXTS = ['.jpg', '.png']


def get_image_list(image_root):
    image_root = pathlib.Path(image_root)
    image_list = list()
    for image_path in image_root.iterdir():
        if image_path.exists() and image_path.suffix.lower() in ACCEPTED_IMAGE_EXTS:
            image_list.append(image_path)
    image_list = sorted(image_list, key = lambda x: int(x.name.split('.')[0].split('_')[1]))
    return image_list

def main():
    parser = ArgumentParser()
    parser.add_argument("--image_root", required=False, type=str, default = './data')
    parser.add_argument("--faiss_bin_path", required=False, type=str, default='building.index.bin')
    parser.add_argument("--device", required=False, type=str, default='cuda:0')
    parser.add_argument("--test_image_path", required=False, type=str, default='./data/worcester_000054.jpg')
    parser.add_argument("--top_k", required=False, type=int, default=11)


    args = parser.parse_args()
    device = torch.device(args.device)

    extractor = MyVGG16(device)

    img_list = get_image_list(args.image_root)

    transform = get_transformation()

    test_image_path = pathlib.Path(args.test_image_path)
    pil_image = Image.open(test_image_path)
    pil_image = pil_image.convert('RGB')
    image_tensor = transform(pil_image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    feat = extractor.extract_features(image_tensor)

    indexer = faiss.read_index(args.faiss_bin_path)
    
    _, indices = indexer.search(feat, k=args.top_k)
    print(indices)
    indices = indices[0]
    for index in indices:
        print(img_list[index])

if __name__ == '__main__':
    main()