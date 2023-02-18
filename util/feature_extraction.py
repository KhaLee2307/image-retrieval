import torch
import torchvision.models as models


def feature_extractor():

    model = models.vgg16(weights='IMAGENET1K_FEATURES')

    return model.eval()

def extract_features(model, image_tensor):
    # Đưa hình ảnh vào mô hình để trích xuất đặc trưng
    with torch.no_grad():
        feature = model.features(image_tensor)
        feature = model.avgpool(feature)
        feature = torch.flatten(feature, start_dim=1)

    # Đưa đặc trưng về dạng numpy array
    return feature.cpu().detach().numpy()