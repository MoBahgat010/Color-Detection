import cv2 as cv
import numpy as np


            # # blank = np.zeros((500,500, 3), dtype='uint8')
            # # cv.rectangle(blank, (0,0), (250,250),(205,155,78), thickness=cv.FILLED)
            # # cv.imshow('Rectangle', blank)

            # img = cv.imread('eminem.jpg')

            # # BGR -> gray scale
            # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # # cv.imshow('Gray', gray)

            # # Blurring
            # blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
            # # cv.imshow('Blur', blur)

            # # Canny Edge
            # canny = cv.Canny(gray, 125, 175)
            # cv.imshow("Canny Edges", canny)

            # # Threshold
            # ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
            # cv.imshow("Binary Image", thresh)

            # contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            # print(f"{len(contours)} Contours found")

            # blank = np.zeros(img.shape, dtype='uint8')
            # cv.drawContours(blank, contours, -1, (0,0,255), 1)
            # cv.imshow('Contours Drawn', blank)


img = cv.imread('eminem.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# BGR -> HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow('HSV', hsv)

# BGR -> LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# cv.imshow('LAB', lab)

# capture = cv.VideoCapture("eminem-stan.mp4")
# while True:
#     isTrue, frame = capture.read()
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     canny = cv.Canny(gray, 150, 255)
#     blank = np.zeros((500,500,3), dtype='uint8')
#     contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
#     cv.drawContours(blank, contours, -1, (255,255,0), 1)
#     cv.imshow("video", blank)

#     if cv.waitKey(20) and 0xFF==ord('d'):
#         break
# capture.release()
# cv.destroyAllWindows()

cv.imshow("Eminem", img)

cv.waitKey(0)