import cv2 as cv
import numpy as np

def get_limits(bgr_value):   
    c = np.uint8([[bgr_value]]) 
    hsvC = cv.cvtColor(c, cv.COLOR_BGR2HSV)
        
    lower_limit = hsvC[0][0][0] - 10, 100, 100
    upper_limit = hsvC[0][0][0] + 10, 255, 255

    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)
    
    return (lower_limit, upper_limit)

colors = [255, 255, 255]

ok = True
capture = cv.VideoCapture(0)
while ok:
    ok, frame = capture.read()
    
    # binr = cv.threshold(frame, 0, 255, cv.THRESH_BINARY)[1] 
    # kernel = np.ones((5, 5), np.uint8) 
    # invert = cv.bitwise_not(binr) 
    # erosion = cv.erode(invert, kernel, iterations=1) 

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_limit = np.array([0,0,140])
    upper_limit = np.array([255,50,255])

    mask = cv.inRange(hsv, lower_limit, upper_limit)

    # mask = cv.GaussianBlur(mask, (15, 15), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    # mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.erode(mask, kernel, iterations=3)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.imshow('white', mask)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w > 50 and h > 50:
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), colors, thickness=2)
            # cv.putText(frame, color_name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[color_name], 2)
            print(f"object at: x={x}, y={y}, width={w}, height={h}")

    cv.imshow("frame", frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()