import pytesseract
import cv2
import AGlicenseplatedetect.convolutions_local as con
from keras.models import load_model
from keras import backend as K


def reco_char(roi):
    config = "-l eng --oem 1 --psm 7"
    # gray = con.convolveCall(roi, "blurring")
    text = pytesseract.image_to_string(roi, config=config)
    print(text)


def cnn_rec(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    alpha_label = {"A": 10,
                   "B": 11,
                   "C": 12,
                   "D": 13,
                   "E": 14,
                   "F": 15,
                   "G": 16,
                   "H": 17,
                   "I": 18,
                   "J": 19,
                   "K": 20,
                   "L": 21,
                   "M": 22,
                   "N": 23,
                   "O": 24,
                   "P": 25,
                   "Q": 26,
                   "R": 27,
                   "S": 28,
                   "T": 29,
                   "U": 30,
                   "V": 31,
                   "W": 32,
                   "X": 33,
                   "Y": 34,
                   "Z": 35}
    model = load_model('model/AZ_09_v1.h5')
    img = cv2.resize(img, (28, 48))

    # cv2.imshow("a", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    img = img.reshape((1, 48, 28, 1))
    img = img.astype("float32")
    img = img / 255.0
    cls = model.predict_classes(img)[0]
    K.clear_session()
    # print("Value ------> ", cls)
    # print("Ind Percentage : ", model.predict(img)[0][cls] * 100)
    if cls < 10:
        return str(cls)
    else:
        for alpha, value in alpha_label.items():
            if str(value) == str(cls):
                return alpha
