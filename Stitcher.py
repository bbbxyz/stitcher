import cv2
import numpy as np
import glob
import sys
import itertools
import os

class Stitcher():
    """
    Stitcher
    Uses BRISK feature detector and brute-force feature matcher
    """

    def __init__(self):
        self.detector = cv2.BRISK_create(thresh=30, octaves = 3 )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def blend(self, img1, img2):
        """
        Blends two images by taking the maximum value across images at each pixel
        """
        out = cv2.max(img1,img2)
        #out = np.median(np.array([ img1, img2 ]), axis=0 )
        return out

    def alignandblend(self, img1, img2, outdims, ratio=0.7):
        """
        Finds a homography between two images using
        matching features. Applies homography and blends the images.

        ratio: used for Lowe's tests on matches
        """
        kp1, des1 = self.detector.detectAndCompute(img1,None)
        kp2, des2 = self.detector.detectAndCompute(img2,None)
        matches = self.matcher.knnMatch(des2,des1, k=2)

        goodmatches = []
        for m,n in matches:
            if( m.distance < n.distance * ratio ):
                goodmatches.append(m)

        # increase ratio if not enough good matches
        if len(goodmatches)<4 :
            if ratio > 1 :
                return img2
            return self.alignandblend(img1, img2, outdims, ratio+0.05)

        X1=[]
        X2=[]
        for g in goodmatches:
            X1.append(kp2[g.queryIdx].pt)
            X2.append(kp1[g.trainIdx].pt)
        X1 = np.array(X1)
        X2 = np.array(X2)

        # find homography from img1 to img2 using RANSAC
        H1,used = cv2.findHomography(X2.astype(np.float32), X1.astype(np.float32), cv2.RANSAC)

        # return original image if no homography found
        if sum(used) == 0 :
            return img2

        # apply homography to img1
        img3 = cv2.warpPerspective(img1, H1, outdims)

        # blend the two warped img1 with img2
        img3 = self.blend(img2,img3)

        return img3

    def cropzeros(self, img):
        """
        Crops an image to nonzero pixels
        """

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return np.zeros((0,0,3))
        cnt = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(cnt)
        crop = img[y:y + h, x:x + w]
        return crop

    def stitch(self, directory, out):
        """
        Stitch together images found in directory
        Assumes pictures are taken from left to right
        """

        imgs = glob.glob(directory+'*.jpg')
        n = len(imgs)

        if n == 0 :
            print("Directory does not contain images")
            exit(1)

        mid = n//2
        midright = imgs[mid+1:]
        midleft = reversed(imgs[:mid])

        imgm = cv2.imread(imgs[mid])

        # create a new image big enough to fit the panorama
        w,h,_ = imgm.shape
        outh = 2*w
        outw = 5*h
        imgf = np.zeros((outh,outw, 3))
        imgf[outh/4:outh/4+w, outw/2:outw/2+h, :] = imgm[:,:,:]
        imgm=imgf.astype(np.uint8)

        for img in midright:
            imgl = cv2.imread(img)
            if imgl is not None:
                w,h,_ = imgl.shape
                if w>0 and h>0 :
                    print(img)
                    imgm = self.alignandblend( imgl, imgm, (outw,outh))

        for img in midleft:
            imgl = cv2.imread(img)
            if not imgl is  None:
                w, h, _ = imgl.shape
                if w > 0 and h > 0:
                    print(img)
                    imgm = self.alignandblend( imgl, imgm, (outw,outh))

        imgout = self.cropzeros(imgm)

        cv2.imwrite(out,imgout)
        print("done")


def main():
    if len(sys.argv) < 3:
        print("Not enough input arguments")
        print("python Stitcher.py [image directory] [output image]")
        exit(2)
    directory = sys.argv[1]

    # check that directory contains images
    if not os.path.exists(directory):
        print("Image directory does not exist")
        exit(1)

    out = sys.argv[2]
    stitcher = Stitcher()
    stitcher.stitch(directory, out)


if __name__ == "__main__":
    main()
