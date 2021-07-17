# 直接使用 onnx 模型进行推理预测
# 和源文件一样的代码 - 只是用于测试
import time

start = time.time()
import onnxruntime
print("import onnx time = ", time.time() - start)
import numpy as np

from PIL import Image

# 提前 load 模型
load_mode_time = time.time()
onnx_model = 'model_light.onnx'
sess = onnxruntime.InferenceSession(onnx_model, None)
print("load_model_time", time.time() - load_mode_time)

core_count_time = 0


def allTogether():
    img_path = "target.jpg"
    # with open(img_path, "rb") as f:
    #     img = f.read()

    img_size = 64
    image = np.array(Image.open(img_path).resize((img_size, img_size)))
    x = image.reshape(1, 64, 64, 3).astype('float32')

    # print("The Following Process is Prediction!")
    # # runtime prediction

    x = x if isinstance(x, list) else [x]
    feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])

    core_start = time.time()
    pred_onnx = sess.run(None, feed)
    global core_count_time
    core_count_time += time.time() - core_start
    # print("predict time = {}".format(time.time() - start))

    # print("\n", pred_onnx)
    result = {
        'cat': pred_onnx[0][0][0],
        'dog': pred_onnx[0][0][1]
    }
    # print(result)
    # print("Done!")


if __name__ == '__main__':
    didi_time = time.time()
    for i in range(10000):
        allTogether()
    all_time = time.time() - didi_time
    print("aaaaa", all_time)
    print("bbbb", core_count_time)
