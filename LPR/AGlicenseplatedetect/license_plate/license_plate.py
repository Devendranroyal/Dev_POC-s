# import the necessary packages
from typing import Any, Union

import numpy as np
import cv2
import imutils
from collections import namedtuple
import AGlicenseplatedetect.convolutions_local as con
import AGlicenseplatedetect.rlsa as rlsa
import AGlicenseplatedetect.PossibleChar as PossibleChar
import AGlicenseplatedetect.Reco as reco
from imutils import perspective
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
import math
import os

hello = 0


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if not any(np.array_equal(x, a) for a in unique_list):
            unique_list.append(x)
    return unique_list


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def slope(p1, p2):
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    return m


def joinBox(boxlist):
    joinedBox = []
    for i in range(len(boxlist)):
        b1 = boxlist[i]
        for j in range(i, len(boxlist)):
            b2 = boxlist[j]
            if np.array_equal(b1, b2):
                pass
            else:
                if b1[1][0] < b2[1][0]:
                    left = b1
                    right = b2
                else:
                    left = b2
                    right = b1
                d1 = distance(left[2], right[1])
                d2 = distance(left[3], right[0])
                if abs(d2 - d1) < 1 and d1 < 15:
                    print(np.array([left[0], left[1], right[2], right[3]]))
                    joinedBox.append(np.array([left[0], left[1], right[2], right[3]]))
                    joinedBox.append(left)
                    joinedBox.append(right)
    return joinedBox


def PolygonArea(corners):
    n = len(corners)  # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


# define the named tupled to store the license plate
LicensePlate = namedtuple("LicensePlateRegion", ["success", "plate", "thresh", "candidates", "reading"])


class LicensePlateDetector:
    def __init__(self, image, minPlateW=60, minPlateH=20):
        # store the image to detect license plates in and the minimum width and height of the
        # license plate region
        self.image = image
        self.minPlateW = minPlateW
        self.minPlateH = minPlateH

    def detect(self):
        # detect and return the license plate regions in the image
        lpRegions = self.detectPlates()

        # loop over the license plate regions
        for lpRegion in lpRegions:
            # detect character candidates in the current license plate region
            lp = self.detectCharacterCandidates(lpRegion)

            # only continue if characters were successfully detected
            if lp.success:
                # yield a tuple of the license plate object and bounding box
                yield (lp, lpRegion)

    def detectPlates(self):
        # initialize the rectangular and square kernels to be applied to the image,
        # then initialize 6the list of license plate regions
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        regions = []
        temp_regions = []

        # Naman's changes to the image

        # convert the image to grayscale, and apply the blackhat operation
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) # Image is gr scaled here
        gray = con.convolveCall(gray, "blurring")
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
        # cv2.imshow("Blackhat", blackhat)
        # soble_img = con.convolveCall(blackhat)
        # canny = cv2.Canny(soble_img, 225, 250)
        # dilated_canny = cv2.dilate(canny, None, iterations=2)
        # eroded_canny = cv2.erode(dilated_canny, None, iterations=1)
        # cv2.imshow("canny", eroded_canny)

        # find regions in the image that are light
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
        light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow("Light", light)

        # compute the Scharr gradient representation of the blackhat image in the x-direction,
        # and scale the resulting image into the range [0, 255]
        gradX = cv2.Sobel(blackhat,
                          ddepth=cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F,
                          dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        # cv2.imshow("Gy", gradX)

        # blur the gradient representation, apply a closing operating, and threshold the
        # image using Otsu's method
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # cv2.imshow("Thresh", thresh)

        # perform a series of erosions and dilations on the image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        # cv2.imshow("E&D", thresh)

        # take the bitwise 'and' between the 'light' regions of the image, then perform
        # another series of erosions and dilations
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=3)
        thresh = cv2.erode(thresh, None, iterations=1)
        # cv2.imshow("Bitwise AND, E&D", thresh)

        # Kuch Toofani
        mask = np.ones(thresh.shape[:2], dtype="uint8") * 255  # create blank
        x, y = mask.shape
        value = max(math.ceil(x / 100), math.ceil(y / 100)) + 20  # heuristic
        mask = rlsa.rlsa(thresh, True, False, value)  # rlsa application
        thresh = cv2.dilate(mask, None, iterations=2)
        thresh = con.convolveCall(thresh, "horizontal_merge")
        # cv2.imshow("Toofaani", thresh)

        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # grab the bounding box associated with the contour and compute the area and
            # aspect ratio
            (w, h) = cv2.boundingRect(c)[2:]
            aspectRatio: Union[float, Any] = w / float(h)

            # compute the rotated bounding box of the region
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.cv.BoxPoints(rect)) if imutils.is_cv2() else cv2.boxPoints(rect)

            if PolygonArea(box) > 1000 and aspectRatio > 1:
                # print("aspectRatio ---> ", aspectRat
                # print("PolygonArea(box) ---> ", PolygonArea(box))
                # lpBox = np.array(box).reshape((-1, 1, 2)).astype(np.int32)
                # print("lpbox ---> ", lpBox)
                # cv2.drawContours(self.image, [lpBox], -1, (0, 255, 0), 2)
                # cv2.imshow("hero image", self.image.copy())
                # print("box ---> ", box)
                # cv2.waitKey(0)
                temp_regions.append(box)

            # ensure the aspect ratio, width, and height of the bounding box fall within
            # tolerable limits, then update the list of license plate regions
            # and h > self.minPlateH and w > self.minPlateW
            if (2 < aspectRatio < 10) and (1000 < PolygonArea(box) < 50000):
                regions.append(box)

        joinedboxes = joinBox(temp_regions)
        if len(joinedboxes) > 0:
            # print("regions + joinedboxes --> ", (regions + joinedboxes))
            regions = unique(regions + joinedboxes)
        # return the list of license plate regions
        return regions

    def detectCharacterCandidates(self, region):
        # apply a 4-point transform to extract the license plate
        plate = perspective.four_point_transform(self.image, region)
        # cv2.imshow("Perspective Transform", imutils.resize(plate, width=400))

        # extract the Value component from the HSV color space and apply adaptive thresholding
        # to reveal the characters on the license plate
        V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 29, offset=15, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)
        listOfMatchingChars = []

        # resize the license plate region to a canonical size
        plate = imutils.resize(plate, width=400)
        thresh = imutils.resize(thresh, width=400)
        # cv2.imshow("Thresh", thresh)

        # perform a connected components analysis and initialize the mask to store the locations
        # of the character candidates
        labels = measure.label(thresh, neighbors=8, background=0)
        charCandidates = np.zeros(thresh.shape, dtype="uint8")

        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue

            # otherwise, construct the label mask to display only connected components for the
            # current label, then find contours in the label mask
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0]  # if imutils.is_cv2() else cnts[1]

            # ensure at least one contour was found in the mask
            if len(cnts) > 0:
                # grab the largest contour which corresponds to the component in the mask, then
                # grab the bounding box for the contour
                c = max(cnts, key=cv2.contourArea)
                (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

                # compute the aspect ratio, solidity, and height ratio for the component
                aspectRatio = boxW / float(boxH)
                solidity = cv2.contourArea(c) / float(boxW * boxH)
                heightRatio = boxH / float(plate.shape[0])

                # determine if the aspect ratio, solidity, and height of the contour pass
                # the rules tests
                keepAspectRatio = aspectRatio < 1.0
                keepSolidity = solidity > 0.15
                keepHeight = heightRatio > 0.4 and heightRatio < 0.95

                # check to see if the component passes all the tests
                if keepAspectRatio and keepSolidity and keepHeight:
                    # Edits start from here
                    listOfMatchingChars.append(PossibleChar.PossibleChar(c))
                    # compute the convex hull of the contour and draw it on the character
                    # candidates mask
                    hull = cv2.convexHull(c)
                    cv2.drawContours(charCandidates, [hull], -1, 255, -1)
            # clear pixels that touch the borders of the character candidates mask and detect
            # contours in the candidates mask
            charCandidates = segmentation.clear_border(charCandidates)
            listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.intCenterX)

        count = 1
        global hello
        craft_plate = np.zeros((80, 50), np.uint8)
        plate_info = ""
        for currentChar in listOfMatchingChars:
            imgROI = thresh[
                     currentChar.intBoundingRectY: currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                     currentChar.intBoundingRectX: currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

            bordersize = 10
            invImg = imgROI.copy()
            invImg = cv2.bitwise_not(invImg)
            invImg = cv2.copyMakeBorder(invImg,
                                        top=bordersize,
                                        bottom=bordersize,
                                        left=bordersize,
                                        right=bordersize,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=255)
            invImg = cv2.bitwise_not(invImg)
            # cv2.imshow("char at {}".format(count), invImg)
            count += 1
            ch = reco.cnn_rec(invImg)
            plate_info = plate_info + ch

        # cv2.imshow("Perspective Transform", imutils.resize(plate, width=400))
        # print("plate_info --> ", plate_info)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # TODO:
        # There will be times when we detect more than the desired number of characters --
        # it would be wise to apply a method to 'prune' the unwanted characters

        # return the license plate region object containing the license plate, the thresholded
        # license plate, and the character candidates
        return LicensePlate(success=True, plate=plate, thresh=thresh,
                            candidates=charCandidates, reading=plate_info)
