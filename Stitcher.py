import cv2
import numpy as np
import glob
import sys


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

    def alignandblend(self, img1, img2, outdims, ratio = 0.75):
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
            return self.alignandblend(img1, img2, outdims, ratio+0.02)

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
        _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return np.zeros((0,0,3))
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = img[y:y + h, x:x + w]
        return crop

    def stitch(self, directory, out):
        """
        Stitch together images found in directory
        Assumes pictures are taken from left to right
        Saves panorama as 'out.jpg'
        """

        imgs = glob.glob(directory+'*.jpg')
        n = len(imgs)
        mid = n//2
        midright = imgs[mid+1:]
        midleft = reversed(imgs[:mid])

        imgm = cv2.imread(imgs[mid])

        # create a new image big enough to fit the panorama
        w,h,_ = imgm.shape
        outh = 2*w
        outw = 8*h
        imgf = np.zeros((outh,outw, 3))
        imgf[outh/4:outh/4+w, outw/2:outw/2+h, :] = imgm[:,:,:]
        imgm=imgf.astype(np.uint8)

        for img in midright:
            print(img)
            imgl = cv2.imread(img)
            imgm = self.alignandblend( imgl, imgm, (outw,outh), 0.6 )

        for img in midleft:
            print(img)
            imgl = cv2.imread(img)
            imgm = self.alignandblend( imgl, imgm, (outw,outh), 0.6 )

        imgout = self.cropzeros(imgm)

        cv2.imwrite(out,imgout)
        print("done")


def main():
    if len(sys.argv) < 3:
        print("Not enough input arguments")
        print("python Stitcher.py [image directory] [output image]")
        exit()
    dir = sys.argv[1]
    out = sys.argv[2]
    stitcher = Stitcher()
    stitcher.stitch(dir, out)


if __name__ == "__main__":
    main()
