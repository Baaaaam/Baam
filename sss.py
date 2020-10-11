import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras


def load_image(file_path):
    return cv2.imread(file_path)

def extract_label(file_name):     #train images의 이름에 따라 2,1,0으로 labeling 하는 function
    if "can" in file_name:
      return 2
    if "glass" in file_name:
      return 1
    else:
      return 0

train_path = "./obj/"
image_files = os.listdir(train_path) #디렉토리에 있는 이미지 파일들을 리스트화
train_images = [load_image(train_path + file) for file in image_files] #데이타 사진프레임의 개체를 opencv라이브러리를 사용하여 images를 read하고 data frame리스트화
train_labels = [extract_label(file) for file in image_files] #labelling set으로 리스트화

train_images = np.array(train_images)
#train_labels = np.arrya(train_labels)

def preprocess_image(img, side=96):            
    min_side = min(img.shape[0], img.shape[1]) 
    img = img[:min_side, :min_side] 
    img = cv2.resize(img, (side,side)) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    return img / 255.0

preview_index = 1
plt.subplot(1,2,1)
plt.imshow(train_images[preview_index])
plt.subplot(1,2,2)
plt.imshow(preprocess_image(train_images[preview_index]), cmap="gray")
plt.show()

print(train_images.shape)


test_path = "./test/"
test_files = os.listdir(test_path)

eval_images = [preprocess_image(load_image(file)) for file in test_files]
eval_model = tf.keras.Sequential([
                          keras.layers.Flatten(input_shape=(96,96)),
                          keras.layers.Dense(50, activation="relu"),
                          keras.layers.Dense(3, activation="softmax")
])

eval_model.load_weights("model.h5") #이걸로 학습된 모델을 

eval_images = np.expand_dims(eval_images, axis=0)
eval_images = np.squeeze(eval_images, axis=0)
print(eval_images.shape)
eval_predictions= eval_model.predict(eval_images)




import RPi.GPIO as GPIO # RPi.GPIO에 정의된 기능을 GPIO라는 명칭으로 사용
import time              # time 모듈

pin = 18                 # Python(BCM)  18번 핀 사용

GPIO.setmode(GPIO.BCM)
GPIO.setup(pin,GPIO.OUT)

p = GPIO.PWM(pin,50)     #PWM 펄스폭 변조, 크기를 50으로 지정
p.start(0)


            
var = prediction[0]

if var == 0 :   
      print ("Right")
      p.ChangeDutyCycle(5)     #Dutycycle을 인자로 받아 실행 중에
      time.sleep(1)
elif var == 1 :  
      print ("Left")
      p.ChangeDutyCycle(10)
      time.sleep(1)
elif var == 2:  
      print ("Center")
      p.ChangeDutyCycle(7.5)
      time.sleep(1)


