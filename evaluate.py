import time
from argparse import ArgumentParser

from src.compute import compute_mAP

ACCEPTED_IMAGE_EXTS = ['.jpg', '.png']


def main():
    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", required=True, type=str, default='VGG16')
    parser.add_argument("--crop", required=False, type=bool, default=False)

    print('Ranking .......')
    start = time.time()

    args = parser.parse_args()

    MAP = compute_mAP(args.feature_extractor, args.crop)

    print(MAP)

    end = time.time()
    print('Finish in ' + str(end - start) + ' seconds')

if __name__ == '__main__':
    main()