# 原来版本使用keras预测的代码
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

from io import BytesIO
import numpy as np
from PIL import Image

from skimage import io
import sys, os, json, time
import urllib

# test deploy update
start = time.time()
from keras.models import model_from_json

print("import keras time = ", time.time() - start)

model = None
# Getting model
model_path = ""
core_count_time = 0

curr_time = time.time()
with open(model_path + 'model.json', 'r') as f:
    model_content = f.read()
    model = model_from_json(model_content)
    # Getting weights
    print(model_path + "weights.h5")
    model.load_weights(model_path + "weights.h5")
    print("loading successfully!")
print("load model time = ", time.time() - curr_time)


def test():
    print("into-test")
    # img_url="https://ae01.alicdn.com/kf/H63d0e76fa59248e0acf6cc1be32410c77.png"
    img_url = "https://dd-static.jd.com/ddimg/jfs/t1/162905/7/17846/76460/6075526dE6af30fb3/b8ed4431cb5c628a.jpg"
    # filename="target_{}.jpg".format(str(time.time()))
    URL = ["https://dd-static.jd.com/ddimg/jfs/t1/162905/7/17846/76460/6075526dE6af30fb3/b8ed4431cb5c628a.jpg",
           "https://dd-static.jd.com/ddimg/jfs/t1/161326/25/18134/37954/60756247E9f3b4458/db669fb5ec68d6bc.jpg"]
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) \
        AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/35.0.1916.114 Safari/537.36',
        'Cookie': 'AspxAutoDetectCookieSupport=1'
    }
    for i in range(2):
        # req = urllib.request.Request(url=img_url, headers=header)
        req = urllib.request.Request(url=URL[i], headers=header)
        response = urllib.request.urlopen(req)
        content = response.read()
        # with open(filename,"wb") as f:
        #     f.write(content)
        response.close()

        # environ = request.environ
        # #print(environ)
        result = predict(content, "")
        print(result)
    return result


def imgUrl_predict(imgurl):
    print("into-url-func")
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) \
        AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/35.0.1916.114 Safari/537.36',
        'Cookie': 'AspxAutoDetectCookieSupport=1'
    }
    req = urllib.request.Request(url=imgurl, headers=header)
    response = urllib.request.urlopen(req)
    content = response.read()
    response.close()

    result = predict(content, "")

    print(result)

    return {"result": result}
    # return TEMPLATE.replace('{fc-result}', result)


from keras.preprocessing import image


def predict(event, context):
    # start = time.time()
    img_size = 64
    img_buffer = np.asarray(bytearray(event), dtype='uint8')
    img = io.imread(BytesIO(img_buffer))
    img = np.array(Image.fromarray(img).resize((img_size, img_size)))
    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img

    global model

    # start_closest = time.time()
    core_time = time.time()
    Y = model.predict(X)
    global core_count_time
    core_count_time += time.time() - core_time
    # print("predict shortest time = {}".format(time.time() - start_closest))

    # print(Y)
    result = {
        'cat': Y[0][0],
        'dog': Y[0][1]
    }
    Y = np.argmax(Y, axis=1)
    classification = 'cat' if Y[0] == 0 else 'dog'
    # print("predict time = {}".format(time.time() - start))
    # print('It is a ' + classification + ' !')
    return result


# 原来的预测数据
# if __name__ == '__main__':
#     img_path = "target.jpg"
#     with open(img_path, "rb") as f:
#         print(predict(f.read(), ""))


if __name__ == '__main__':

    img_path = "target.jpg"
    with open(img_path, "rb") as f:
        predict(f.read(), "")
    didi_time = time.time()
    # for i in range(10000):
    #     with open(img_path, "rb") as f:
    #         predict(f.read(), "")
    # all_time = time.time() - didi_time
    # print("aaaaa", all_time)
    # print("bbbb",core_count_time)
