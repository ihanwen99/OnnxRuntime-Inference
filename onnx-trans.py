# 从keras模型转换成新模型的代码
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from keras.models import model_from_json
import keras2onnx
import onnxruntime
import onnx

from io import BytesIO
import numpy as np
from PIL import Image
from skimage import io

model = None
# Getting model
model_path = ""

with open(model_path + 'model.json', 'r') as f:
    model_content = f.read()
    model = model_from_json(model_content)
    # Getting weights
    print(model_path + "weights.h5")
    model.load_weights(model_path + "weights.h5")
    print("loading successfully!")

print("\n\n\nThe Following Process is Converting!")
# convert to onnx model
onnx_model = keras2onnx.convert_keras(model, model.name)

print("\n\n\nThe Following Process is Saving!")
temp_model_file = './model_light.onnx'
onnx.save_model(onnx_model, temp_model_file)

img_path="target.jpg"
with open(img_path, "rb") as f:
    img=f.read()

img_size = 64
image = np.array(Image.open(img_path).resize((img_size,img_size)))
x=image.reshape(1,64,64,3).astype('float32')

# 以下是我注释的内容
# img_buffer = np.asarray(bytearray(img), dtype='uint8')
# img = io.imread(BytesIO(img_buffer))
# img = np.array(Image.fromarray(img).resize((img_size, img_size)))
# x = np.zeros((1, 64, 64, 3), dtype='float32')
# x[0] = img

print("\n\n\nThe Following Process is Prediction!")
# # runtime prediction
content = onnx_model.SerializeToString()
sess = onnxruntime.InferenceSession(content)
x = x if isinstance(x, list) else [x]
feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
pred_onnx = sess.run(None, feed)
print(pred_onnx)
print("Done!")

