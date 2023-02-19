import torch
from torchvision import transforms
import torchvision.models as models
import numpy as np
import cv2


class MyVGG16(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.model = models.vgg16(weights='IMAGENET1K_FEATURES')
        self.model = self.model.eval()
        self.model = self.model.to(device)
        self.shape = 25088

    def extract_features(self, image):
        transform = transforms.Compose([transforms.Normalize(mean=[0.48235, 0.45882, 0.40784], 
                                    std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])])
        image = transform(image)

        # Đưa hình ảnh vào mô hình để trích xuất đặc trưng
        with torch.no_grad():
            feature = self.model.features(image)
            feature = self.model.avgpool(feature)
            feature = torch.flatten(feature, start_dim=1)

        # Đưa đặc trưng về dạng numpy array
        return feature.cpu().detach().numpy()

class RGBHistogram():
    def __init__(self, device):
        pass
        self.shape = 672

    def extract_features(self, image):
        image = image.cpu().numpy()
        features = []
        for img in image:
            img *= 255
            # Tính toán histogram của từng kênh màu
            hist_red = cv2.calcHist([img], [0], None, [224], [0, 224])
            hist_green = cv2.calcHist([img], [1], None, [224], [0, 224])
            hist_blue = cv2.calcHist([img], [2], None, [224], [0, 224])

            # Chuẩn hóa histogram
            cv2.normalize(hist_red, hist_red, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_green, hist_green, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_blue, hist_blue, 0, 1, cv2.NORM_MINMAX)
            
            # Gộp các histogram của các kênh màu thành một feature vector
            feature_vector = np.concatenate((hist_red, hist_green, hist_blue))
            feature_vector.resize(len(feature_vector))
            features.append(feature_vector)
        return np.array(features)