import numpy as np
import cv2
from skimage import color
import os

objPixel=None
backPixel=None

names=['6pm_daylight','1pm_day_light','dim_light','bright_light']
night_light=cv2.imread('Original_images/6pm_daylight.jpg',cv2.COLOR_BGR2RGB)
day_light=cv2.imread('Original_images/1pm_day_light.jpg',cv2.COLOR_BGR2RGB)
dim_light=cv2.imread('Original_images/dim_light.jpg',cv2.COLOR_BGR2RGB)
bright_light=cv2.imread('Original_images/bright_light.jpg',cv2.COLOR_BGR2RGB)
origImages=[bright_light,day_light,dim_light,bright_light]

def toGrayscale(img):
    grayImg=np.matmul(img,[[0.299],[0.587],[0.114]])
    grayImg=np.round(grayImg,0)
    grayImg=grayImg.astype("uint8")
    return grayImg


def discretize(img,n):
    discImg=(img//(256/n))*256/n
    return discImg

def toBinary(img):
    th,binartImage = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th, binartImage

def changeHSV(img):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsvImg2=hsvImg.copy()
    hsvImg2[:,:,0]=(hsvImg2[:,:,0]+10)%256
    hsvImg2[:,:,1]=(hsvImg2[:,:,1]+20)%256
    hsvImg2[:,:,2]=(hsvImg2[:,:,2]+30)%256
    rgbImg = cv2.cvtColor(hsvImg2, cv2.COLOR_HSV2RGB)
    return rgbImg

def mouse_callback1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the first point
        global objPixel,backPixel
        if objPixel is None:
            objPixel = (x, y)
        # Store the second point
        elif backPixel is None:
            backPixel = (x, y)
        cv2.putText(bright_light_c, f'({x},{y})',(x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

# converting RGB image to Grayscale
grayImages=[]
for i,img in enumerate(origImages):
    grayImg=toGrayscale(img)
    grayImages.append(grayImg)
    if not os.path.exists('Gray_images'):
        os.makedirs('Gray_images')
    cv2.imwrite('Gray_images/gray_{0}.jpg'.format(names[i]),grayImg)

#discretizing Grayscale image
for i,img in enumerate(grayImages):
    discImg=discretize(img,4)
    if not os.path.exists('Discretized_images'):
        os.makedirs('Discretized_images')
    cv2.imwrite('Discretized_images/discret_{0}.jpg'.format(names[i]),discImg)

#binarization of Grayscale image
for i,img in enumerate(grayImages):
    th,binaryImg=toBinary(img)
    if not os.path.exists('Binary_images'):
        os.makedirs('Binary_images')
    cv2.imwrite('Binary_images/binary_{0}.jpg'.format(names[i]),binaryImg)
    print('Binarization threshold for gray_{0}.jpg is {1}'.format(names[i],th))
    
#Changing Hue Saturation and Brightness of image
for i,img in enumerate(origImages):
    hsvChangedImg=changeHSV(img)
    if not os.path.exists('HSVchanged_images'):
        os.makedirs('HSVchanged_images')
    cv2.imwrite('HSVchanged_images/hsv_changed_{0}.jpg'.format(names[i]),hsvChangedImg)


# Create a window to display the image
cv2.namedWindow('Image')
#Copying image in order not to contaminate original image 
bright_light_c=bright_light.copy()
cv2.setMouseCallback("Image", mouse_callback1)

while True:
   cv2.imshow('Image',bright_light_c)
   k = cv2.waitKey(1) & 0xFF  #ESC button
   if k == 27:
      break
cv2.destroyAllWindows()



print('pixel1={0}-{1}'.format(objPixel[0],objPixel[1]))
print(bright_light[objPixel[0],objPixel[1],0:3])

print('pixel2={0}-{1}'.format(backPixel[0],backPixel[1]))
print(bright_light[backPixel[0],backPixel[1],0:3])
# pixel1 is on object, pixel2 is on background
pixel1=bright_light[objPixel[0],objPixel[1],0:3].reshape(1,1,-1)
pixel2=bright_light[backPixel[0],backPixel[1],0:3].reshape(1,1,-1)

#Converting images and pixels to lab format
lab1 = color.rgb2lab(bright_light)
lab2=color.rgb2lab(pixel1)
dE = color.deltaE_ciede2000(lab1,lab2)
if not os.path.exists('Closeness'):
    os.makedirs('Closeness')
cv2.imwrite('Closeness/close_{0}.jpg'.format(names[0]),dE)


lab1 = color.rgb2lab(bright_light)
lab2=color.rgb2lab(pixel2)
dE2 = color.deltaE_ciede2000(lab1,lab2)
if not os.path.exists('Closeness'):
    os.makedirs('Closeness')
cv2.imwrite('Closeness/close_back{0}.jpg'.format(names[0]),dE2)