import cv2
import matplotlib.pyplot as plt
import numpy as np
def nothing(x):
    pass

img3=cv2.imread("robert.png")
    img3 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernelX=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernelY=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    prwX=cv2.filter2D(img3,-1,kernelX)
    prwY=cv2.filter2D(img3,-1,kernelY)
    kenarli=prwX+prwY
    cv2.imshow("kenarli",kenarli)