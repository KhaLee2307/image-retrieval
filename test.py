import cv2
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import urllib
import numpy as np

# tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

# Tải mô hình DELF được đào tạo trên tập dữ liệu Landmarks-Clean/Landmarks-Full
module_url = "https://tfhub.dev/google/delf/1"
delf_module = hub.Module("https://tfhub.dev/google/delf/1")

def extract_features(image_paths):
    # Trích xuất deep local features từ từng hình ảnh
    features_list = []
    for path in image_paths:
        # Đọc hình ảnh
        image = tf.image.decode_jpeg(path, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Chuyển đổi sang dạng tensor
        delf_inputs = {
        # An image tensor with dtype float32 and shape [height, width, 3], where
        # height and width are positive integers:
        'image': image,
        # Scaling factors for building the image pyramid as described in the paper:
        'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
        # Image features whose attention score exceeds this threshold will be
        # returned:
        'score_threshold': 100.0,
        # The maximum number of features that should be returned:
        'max_feature_num': 1000,
        }
        # Trích xuất deep local features từ hình ảnh
        image_features = delf_module(delf_inputs, as_dict=True)
        with tf.Session() as sess:
            array = sess.run(image_features['features'])
        print(array)
        break
        # features_list.append(image_features["feature_descriptor"].numpy().flatten())
    
    return features_list

extract_features('my_image.jpg')