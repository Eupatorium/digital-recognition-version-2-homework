import cv2, numpy as np, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as ac
import os, ssl, time
from PIL import Image
import PIL.ImageOps

x = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

xtrain,xtest,ytrain,ytest = tts(x,y,random_state=9, train_size = 7500, test_size=2500)
xps = xtrain/255.0
x_testscale = xtest/255.0
model=LogisticRegression(solver='saga', multi_class='multinomial').fit(xps,ytrain)

ypredict = model.predict(x_testscale)
accuracy = ac(ytest, ypredict)
print(accuracy)

cap = cv2.VideoCapture(0)
while(True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        ul = (int(width/2-56),int(height/2-56))
        bl = (int(width/2+56),int(height/2+56))

        cv2.rectangle(gray,ul,bl,(0,255,0),2)

        aoi = gray[ul[1]:bl[1],ul[0],bl[0]]
        iam_pli = Image.fromarray(aoi)

        ibw = iam_pli.convert('L')
        ibw_resized = ibw.resize((28,28),Image.ANTIALIAS)
        ibw_resized_inverted = PIL.ImageOps.invert(ibw_resized)

        pixel_filter = 20

        minpixel = np.percentile(ibw_resized_inverted, pixel_filter)

        ibw_resized_inverted_scale = np.clip(ibw_resized_inverted-minpixel,0,255)

        maxpixel = np.max(ibw_resized_inverted)

        ibw_resized_inverted_scale = np.asarray(ibw_resized_inverted_scale)/maxpixel

        test_sample = np.array(ibw_resized_inverted_scale).reshape(1,784)

        test_predict = model.predict(test_sample)

        print("predicted class is",test_predict)
        cv2.imshow('frame', gray)
        
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    except Exception as e:
        pass 
cap.release()
cv2.destroyAllWindows()
        
