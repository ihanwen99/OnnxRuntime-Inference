# 直接使用 onnx 模型进行推理预测
import time

start = time.time()
import onnxruntime
print("import onnx time = ", time.time() - start)
import numpy as np

from PIL import Image

onnx_model = 'model_light.onnx'

img_path = "target.jpg"
with open(img_path, "rb") as f:
    img = f.read()

img_size = 64
image = np.array(Image.open(img_path).resize((img_size, img_size)))
x = image.reshape(1, 64, 64, 3).astype('float32')

print("The Following Process is Prediction!")
# # runtime prediction
sess = onnxruntime.InferenceSession(onnx_model, None)
x = x if isinstance(x, list) else [x]
feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])

start = time.time()
pred_onnx = sess.run(None, feed)
print("predict time = {}".format(time.time() - start))

print("\n", pred_onnx)
result = {
    'cat': pred_onnx[0][0][0],
    'dog': pred_onnx[0][0][1]
}
print(result)
print("Done!")
