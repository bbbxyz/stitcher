import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

imgl = cv.imread('images/r.jpg')
imgm = cv.imread('images/m.jpg')
imgr = cv.imread('images/l.jpg')

w,h,_ = imgm.shape
outh = 2*w
outw = 3*h

imgf = np.zeros((outh,outw, 3))
imgf[outh/2:outh/2+w, outw/3:outw/3+h,  :] = imgm[:,:,:]
imgm=imgf.astype(np.uint8)

sift = cv.BRISK_create()

kpl, desl = sift.detectAndCompute(imgl,None)
kpm, desm = sift.detectAndCompute(imgm,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(desm,desl)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
X1=[]
X2=[]
for g in matches[:100]:
    X1.append(kpm[g.queryIdx].pt)
    X2.append(kpl[g.trainIdx].pt)
X1 = np.array(X1)
X2 = np.array(X2)
H1,_ = cv.findHomography(X2.astype(np.float32), X1.astype(np.float32))

kpr, desr = sift.detectAndCompute(imgr,None)
# Match descriptors.
matches = bf.match(desm,desr)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
X1=[]
X2=[]
for g in matches[:100]:
    X1.append(kpm[g.queryIdx].pt)
    X2.append(kpr[g.trainIdx].pt)
X1 = np.array(X1)
X2 = np.array(X2)
H2,_ = cv.findHomography(X2.astype(np.float32), X1.astype(np.float32))

imgrm = cv.warpPerspective(imgr, H2, (outw,outh))
imgm = cv.max(imgm,imgrm)
imglm = cv.warpPerspective(imgl, H1, (outw,outh))
imgm = cv.max(imgm,imglm)

cv.imwrite('stitched.png',imgm)





