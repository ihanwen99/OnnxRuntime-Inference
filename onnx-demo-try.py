# onnx 官方给的代码示例
# 前面的两行是注释掉 tensorflow 的提示信息
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import keras2onnx
import onnxruntime
import onnx

print("\n\n\nThe Following Process is Image Processing!")
# image preprocessing
img_path = 'target.jpg'  # make sure the image is in img_path
img_size = 224
img = image.load_img(img_path, target_size=(img_size, img_size))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# print("\n\n\nThe Following Process is Loading Model!")
# # load keras model
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(include_top=True, weights='imagenet')

# print("\n\n\nThe Following Process is Converting!")
# # convert to onnx model
# onnx_model = keras2onnx.convert_keras(model, model.name)

# print("\n\n\nThe Following Process is Saving!")
# temp_model_file = '/home/hwliu/workspace/network/onnx-inference/model.onnx'
# onnx.save_model(onnx_model, temp_model_file)

print("\n\n\nThe Following Process is Prediction!")
# # runtime prediction
onnx_model = "model.onnx"
sess = onnxruntime.InferenceSession(onnx_model, None)  # 直接载入模型

# 注释掉的是承接的是转换的流程
# content = onnx_model.SerializeToString()
# sess = onnxruntime.InferenceSession(content)
x = x if isinstance(x, list) else [x]
feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
pred_onnx = sess.run(None, feed)
print(pred_onnx)
print("Done!")
