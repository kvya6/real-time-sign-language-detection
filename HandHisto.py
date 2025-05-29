import cv2
import numpy as np
import pickle

def build_squares(img):
    x, y, w, h = 420, 140, 10, 10
    d = 10
    imgCrop = None
    crop = None
    for i in range(10):
        for j in range(5):
            roi = img[y:y+h, x:x+w]
            if imgCrop is None:
                imgCrop = roi
            else:
                imgCrop = np.hstack((imgCrop, roi))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            x += w + d
        if crop is None:
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop))
        imgCrop = None
        x = 420
        y += h + d
    return crop

def get_hand_hist():
    cam = cv2.VideoCapture(1)
    if not cam.read()[0]:
        cam = cv2.VideoCapture(0)

    hist = None
    flagPressedC, flagPressedS = False, False

    print("📷 Press 'c' to capture histogram, 's' to save and exit.")

    while True:
        ret, img = cam.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('c'):
            imgCrop = build_squares(img.copy())
            hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            flagPressedC = True
            print("✅ Histogram captured.")
        elif keypress == ord('s') and flagPressedC:
            with open("hist", "wb") as f:
                pickle.dump(hist, f)
            print("💾 Histogram saved to 'hist'. Exiting...")
            break

        if flagPressedC and hist is not None:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            dst = cv2.filter2D(dst, -1, disc)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            cv2.imshow("Thresholded Hand", thresh)

        if not flagPressedC:
            build_squares(img)

        cv2.imshow("Set Hand Histogram", img)

    cam.release()
    cv2.destroyAllWindows()

get_hand_hist()
