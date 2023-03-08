import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

from skimage.feature import local_binary_pattern


class MyResnet50(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.model = models.resnet50(weights='IMAGENET1K_V2')
        # Get the layers of the model
        self.modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*self.modules)
        self.model = self.model.eval()
        self.model = self.model.to(device)
        self.shape = 2048 # the length of the feature vector

    def extract_features(self, image):
        transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225])])
        image = transform(image)

        # Pass the image through the Resnet50 model and get the feature maps of the input image
        with torch.no_grad():
            feature = self.model(image)
            feature = torch.flatten(feature, start_dim=1)

        # Return features to numpy array
        return feature.cpu().detach().numpy()
        
class MyVGG16(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.model = models.vgg16(weights='IMAGENET1K_FEATURES')
        self.model = self.model.eval()
        self.model = self.model.to(device)
        self.shape = 25088 # the length of the feature vector

    def extract_features(self, image):
        transform = transforms.Compose([transforms.Normalize(mean=[0.48235, 0.45882, 0.40784], 
                                    std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])])
        image = transform(image)

        # Feed the image into the model for feature extraction
        with torch.no_grad():
            feature = self.model.features(image)
            feature = torch.flatten(feature, start_dim=1)

        # Return features to numpy array
        return feature.cpu().detach().numpy()

class RGBHistogram():
    def __init__(self, device):
        self.shape = 768 # the length of the feature vector

    def extract_features(self, image):
        image = image.cpu().numpy()
        features = []
        for img in image:
            # Convert to format when reading images from CV2
            img *= 255
            img = img.reshape(img.shape[1], img.shape[2], img.shape[0])

            # Calculate the histogram of each color channel
            hist_red = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_green = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_blue = cv2.calcHist([img], [2], None, [256], [0, 256])

            # Normalize histogram
            cv2.normalize(hist_red, hist_red, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_green, hist_green, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_blue, hist_blue, 0, 1, cv2.NORM_MINMAX)
            
            # Merge histograms of color channels into a feature vector
            feature_vector = np.concatenate((hist_red, hist_green, hist_blue))
            feature_vector.resize(len(feature_vector))
            features.append(feature_vector)
        return np.array(features)

class LBP():
    def __init__(self, device):
        self.shape = 26 # the length of the feature vector

    def extract_features(self, image):
        n_points = 24
        radius = 3

        image = image.cpu().numpy()
        features = []
        for img in image:
            # Convert to format when reading images from CV2
            img *= 255
            img = img.reshape(img.shape[1], img.shape[2], img.shape[0])

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp = local_binary_pattern(gray, n_points, radius, method="default")
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float32")
            hist /= (hist.sum() + 1e-7)

            features.append(hist)

        return np.array(features)