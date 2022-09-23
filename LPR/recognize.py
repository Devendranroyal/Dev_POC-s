
# import the necessary packages
from __future__ import print_function
from AGlicenseplatedetect.license_plate import LicensePlateDetector
from imutils import paths
import numpy as np
import imutils
import cv2
import pandas as pd


# Creating dataframe
columns = ["Location of image", "License Plate Detected"]
data = []


def alpr_function(imagePath):
    # load the image
    image = cv2.imread(imagePath)
    print(imagePath)

    # if the width is greater than 640 pixels, then resize the image
    if image.shape[1] > 640:
        image = imutils.resize(image, width=720)

    # initialize the license plate detector and detect the license plates and charactors
    lpd = LicensePlateDetector(image)
    plates = lpd.detect()

    rec_plates = []
    # loop over the license plate regions and draw the bounding box surrounding the
    # license plate
    for (i, (lp, lpBox)) in enumerate(plates):
        if len(lp.reading) >= 7:
            lpBox = np.array(lpBox).reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(image, [lpBox], -1, (0, 255, 0), 2)
            rec_plates.append(lp.reading)

    return rec_plates
