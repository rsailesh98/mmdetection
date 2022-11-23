
import json
import sys
import os
import time
import numpy as np
import cv2
import onnx
import onnxruntime
from onnx import numpy_helper


model='yolov3.onnx'

#Preprocess
img = cv2.imread('tests/data/color.jpg')
#img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_AREA)
#img.resize((1, 1, 28, 28))

#Convert image into array f32
data = json.dumps({'data': img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')

#Pass data to model for inference
data = json.dumps({'data': img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')
session = onnxruntime.InferenceSession(model,
                                       providers=['CUDAExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(input_name)
print(output_name)

#Passing the input to the session and print prediction
result = session.run([output_name], {input_name: data})
prediction=int(np.argmax(np.array(result).squeeze(), axis=0))
print(prediction)