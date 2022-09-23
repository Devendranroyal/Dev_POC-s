from skimage.exposure import rescale_intensity
import numpy as np
import cv2

def convolveCall(image, op):

    def convolve(image, kernel):
        # grab the spatial dimensions of the image, along with
        # the spatial dimensions of the kernel
        (iH, iW) = image.shape[:2]
        (kH, kW) = kernel.shape[:2]

        # allocate memory for the output image, taking care to
        # "pad" the borders of the input image so the spatial
        # size (i.e., width and height) are not reduced
        pad = (kW - 1) // 2
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        output = np.zeros((iH, iW), dtype="float32")

        for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                # extract the ROI of the image by extracting the
                # *center* region of the current (x, y)-coordinates
                # dimensions
                roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

                # perform the actual convolution by taking the
                # element-wise multiplicate between the ROI and
                # the kernel, then summing the matrix
                k = (roi * kernel).sum()

                # store the convolved value in the output (x,y)-
                # coordinate of the output image
                output[y - pad, x - pad] = k

        output = rescale_intensity(output, in_range=(0, 255))
        output = (output * 255).astype("uint8")

        # return the output image
        return output


    # construct the Laplacian kernel used to detect edge-like
    # regions of an image
    kernals = {
        "laplacian" : np.array((
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]), dtype="int"),

        "outline" : np.array((
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]), dtype="int"),

        "soble_k" : np.array((
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]), dtype="int"),

        "sharp_the_image" : np.array((
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]), dtype="int"),

        "blurring" : np.array((
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]), dtype="float"),

        "horizontal_merge" : np.array((
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]), dtype="int")
    }

    convoleOutput = convolve(image, kernals[op])
    # opencvOutput = cv2.filter2D(image, -1, laplacian)

    # show the output images
    # cv2.imshow("convole", convoleOutput)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return convoleOutput