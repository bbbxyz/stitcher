import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob


brisk = cv.BRISK_create()
#brisk = cv.xfeatures2d.SURF_create()
# create BFMatcher object

#bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

def blend(img1, img2):
    out = cv.max(img1,img2)
    #out = np.median(np.array([ img1, img2 ]), axis=0 )
    return out


def alignAndBlend(img1, img2, detector,  outdims, ratio = 0.75):
    kp1, des1 = detector.detectAndCompute(img1,None)
    kp2, des2 = detector.detectAndCompute(img2,None)

    # Match descriptors.
    #bf = cv.BFMatcher()
    # FLANN parameters
    index_params = dict(algorithm=6,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des2,des1, k=2)

    goodmatches = []
    for m,n in matches:
        if( m.distance < n.distance * ratio ):
            goodmatches.append(m)
    print(len(goodmatches))

    X1=[]
    X2=[]
    for g in goodmatches:
        X1.append(kp2[g.queryIdx].pt)
        X2.append(kp1[g.trainIdx].pt)
    X1 = np.array(X1)
    X2 = np.array(X2)

    #find homography using RANSAC
    H1,_ = cv.findHomography(X2.astype(np.float32), X1.astype(np.float32), cv.RANSAC)
    img3 = cv.warpPerspective(img1, H1, outdims)

    #blend the two images
    img3 = blend(img2,img3)

    return img3


def cropZeros(frame):
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    _,thresh = cv.threshold(gray,1,255,cv.THRESH_BINARY)
    _, contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.zeros((0,0,3))
    cnt = contours[0]
    x, y, w, h = cv.boundingRect(cnt)
    crop = frame[y:y + h, x:x + w]
    return crop



'''
assume pictures are in order from left to right
stitch mid-right then mid-left
'''


directory='images/room/*.jpg'
imgs = glob.glob(directory)
n = len(imgs)
mid = n//2
midright = imgs[mid:]
midleft = reversed(imgs[:mid])

imgm = cv.imread(imgs[mid])
w,h,_ = imgm.shape
outh = 2*w
outw = 8*h

imgf = np.zeros((outh,outw, 3))
imgf[outh/4:outh/4+w, outw/2:outw/2+h, :] = imgm[:,:,:]
imgm=imgf.astype(np.uint8)




for img in midright:
    print(img)
    imgl = cv.imread(img)
    imgm = alignAndBlend(imgl,imgm, brisk, (outw,outh), 0.65)


for img in midleft:
    print(img)
    imgl = cv.imread(img)
    imgm = alignAndBlend(imgl,imgm, brisk, (outw,outh), 0.65)


imgout = cropZeros(imgm)


cv.imwrite('stitched.png',imgout)
print("done")
#plt.imshow(imgout), plt.show()



