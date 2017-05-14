import cv2
import sys
import Stitcher
import os, shutil
import numpy as np

def extractframes(vid, fskip = 15):
    directory = vid[:-4]+"/"
    os.mkdir(directory)
    vidcap = cv2.VideoCapture( vid )

    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if count%fskip == 0:
         cv2.imwrite(directory+"%4d.jpg" % count, image)  # save frame as JPEG file
        count += 1
    return directory


def main():
    if len(sys.argv) < 3:
        print("Not enough input arguments")
        print("python stitchvideo.py [input video] [output image]")
        exit(2)
    vid = sys.argv[1]
    out = sys.argv[2]

    # check that video exists
    if not os.path.isfile(vid):
        print("Video does not exist")
        exit(1)

    # extract frames from video and store then in temporary dir
    directory = extractframes( vid )

    # stitch frames together
    stitcher = Stitcher.Stitcher()
    stitcher.stitch(directory, out)

    # remove temp. dir
    shutil.rmtree(directory)


if __name__ == "__main__":
    main()
