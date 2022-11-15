import cv2
import numpy as np
def nothing(x):
    pass
img =cv2.imread("writing2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.namedWindow('image')




cv2.createTrackbar('Thmin', 'image', 0, 255, nothing)
cv2.setTrackbarPos('Thmin', 'image', 65)
thmin=0
while(1):
    thmin = cv2.getTrackbarPos('Thmin', 'image')

    _, result = cv2.threshold(img, thmin, 255, cv2.THRESH_BINARY)
    cv2.imshow('image', result)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
