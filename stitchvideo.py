import cv2
import sys
import Stitcher
import os, shutil
import numpy as np

def extractframes(vid, fskip = 15):
    dir = vid[:-4]+"/"
    os.mkdir(dir)
    vidcap = cv2.VideoCapture(vid)

    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if count%fskip == 0:
         cv2.imwrite(dir+"%4d.jpg" % count, image)  # save frame as JPEG file
        count += 1
    return dir


def main():
    if len(sys.argv) < 3:
        print("Not enough input arguments")
        print("python stitchvideo.py [input video] [output image]")
        exit()
    vid = sys.argv[1]
    out = sys.argv[2]

    # extract frames from video and store then in temporary dir
    dir = extractframes( vid )

    # stitch frames together
    stitcher = Stitcher.Stitcher()
    stitcher.stitch(dir, out)

    # remove temp. dir
    shutil.rmtree(dir)


if __name__ == "__main__":
    main()
