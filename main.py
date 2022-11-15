import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import random

img = cv2.imread("writing2.jpg")

roberts_cross_v = np.array( [[ 0, 0, 0 ],
                             [ 0, 1, 0 ],
                             [ 0, 0,-1 ]] )

roberts_cross_h = np.array( [[ 0, 0, 0 ],
                             [ 0, 0, 1 ],
                             [ 0,-1, 0 ]] )

def sekiz_bit(foto):
    s= foto.astype(float)
    s = rescale(s)


    return (s / 256).astype(np.uint8)
def onalti_bit(foto):
    s =cv2.cvtColor(foto, cv2.COLOR_RGB2GRAY)
    s = foto.astype(float)/3
    s = np.uint8(foto)
    s = rescale(s)

    return s
def yirmidort_bit(foto):
    s = foto.astype(float)


    return (s / 256).astype(np.uint8)

def yazi_netleme():
    img = cv2.imread("writing2.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _ ,result = cv2.threshold(img, 107, 255, cv2.THRESH_BINARY)

    return cv2.imshow('Netleştirilmiş', result)


def stack(*args):
    return np.hstack(args)


def rescale(foto):
    s = foto.astype(float)
    s -= np.min(s)
    s /= np.max(s)
    return (s*255).astype(np.uint8)


def fotograf_negatifi(foto):
    L = np.max(foto) # 255
    negatif_foto = L - foto
    return negatif_foto


def log_donusumu(r, c):
    r = r.astype(float)
    s = c*np.log(1 + r)
    s = rescale(s)
    return s
def kuvvet_donusumu(r, c, gamma):
    r = r.astype(float)
    s = c*r**gamma
    s = rescale(s)
    return s


def main():
    foto = cv2.imread("manzara.jpg")
    negatif_foto = fotograf_negatifi(foto)
    log_foto = log_donusumu(foto, c=1)
    sekiz=sekiz_bit(foto)
    onalti=onalti_bit(foto)
    yirmidort=yirmidort_bit(foto)

    c = 1
    gamma_degerleri = [3, 4, 5]
    kuvvet_fotolari = []
    for gamma in gamma_degerleri:
        kuvvet_foto = kuvvet_donusumu(foto, c=c, gamma=gamma)
        kuvvet_fotolari.append(kuvvet_foto)

    satir1 = stack(foto, kuvvet_fotolari[0])
    satir2 = stack(*kuvvet_fotolari[1:])

    grid = np.vstack((satir1, satir2))
    plt.figure(1)
    plt.title("Gamma Dönüşümleri")

    plt.imshow(grid, cmap="gray")
    ################################





    ################################






    yan_yana_negatif = stack(foto, negatif_foto)
    yan_yana_log = stack(foto, log_foto)





    plt.figure(2)
    plt.title("Logaritmik dönüşüm")
    plt.imshow(yan_yana_log,cmap="hsv")

    plt.figure(3)
    plt.title("Negatif dönüşüm")
    plt.imshow(yan_yana_negatif, cmap="hsv")
    plt.figure(4)
    satir11 = stack(sekiz, onalti)
    satir22 = stack(yirmidort, foto)
    grid2 = np.vstack((satir11, satir22))
    plt.title("Bit dönüşümleri")
    plt.imshow(grid2, cmap="Greys")

    ##################SOBEL###################
    image=cv2.imread("sobeltest.jpg")
    sobel = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=3)
    stacked_sobel=stack(image,sobel)
    plt.figure(5)
    plt.title("Sobel")
    plt.imshow(stacked_sobel, cmap="hsv")
    ##################SOBEL###################

    ##################ROBERT###################
    img2=cv2.imread("robert.png")
    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    vertical = ndimage.convolve(img2, roberts_cross_v)
    horizontal = ndimage.convolve(img2, roberts_cross_h)
    edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
    plt.figure(6)
    plt.title("Robert")
    plt.imshow(edged_img, cmap="gray")
    ##################ROBERT###################

    ##################PREWITT###################
    img3=cv2.imread("robert.png")
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)
    kernelX=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernelY=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    prwX=cv2.filter2D(img3,-1,kernelX)
    prwY=cv2.filter2D(img3,-1,kernelY)
    kenarli=prwX+prwY
    stacked_prewitt = stack(img3, kenarli)
    plt.figure(7)
    plt.title("Prewitt")
    plt.imshow(stacked_prewitt, cmap="gray")




    ##################PREWITT###################

    ##################SALT&PAPER###################
    img4 = cv2.imread("lena.png",0)

    img4 = img4 / 255
    x, y = img4.shape
    g = np.zeros((x, y), dtype=np.float32)


    pepper = 0.1
    salt = 0.95


    for i in range(x):
        for j in range(y):
            rdn = np.random.random()
            if rdn < pepper:
                g[i][j] = 0
            elif rdn > salt:
                g[i][j] = 1
            else:
                g[i][j] = img4[i][j]

    cv2.imshow("pepper", g)
    plt.figure(8)
    plt.title("test")
    plt.hist(cv2.imread("lena.png",0).ravel(),256,[0,256])







    ##################SALT&PAPER###################

















    yazi_netleme()


    plt.show()




if __name__ == "__main__":
    main()