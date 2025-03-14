import cv2 as cv
import numpy as np

def get_limits(color_dict):
    limits = {}
    for color_name, bgr_value in color_dict.items():
        if color_name == 'white':
            lower_limit = np.array([0,0,150])
            upper_limit = np.array([255,60,255])
        elif color_name == 'dark_red':
            # Custom HSV range for dark red (low value)
            lower_limit = np.array([0, 100, 50], dtype=np.uint8)
            upper_limit = np.array([10, 255, 255], dtype=np.uint8)
        else:
            c = np.uint8([[bgr_value]])
            hsvC = cv.cvtColor(c, cv.COLOR_BGR2HSV)

            lower_limit = hsvC[0][0][0] - 10, 100, 100
            upper_limit = hsvC[0][0][0] + 10, 255, 255

            lower_limit = np.array(lower_limit, dtype=np.uint8)
            upper_limit = np.array(upper_limit, dtype=np.uint8)

        limits[color_name] = (lower_limit, upper_limit)
    
    return limits

colors = {
    'red': [65, 60, 160],
    'white': [255, 255, 255],
    'blue': [230, 216, 173],
    'green': [80, 220, 90],
    'yellow': [65, 210, 200],
    'orange': [80, 140, 235],
}

ok = True
capture = cv.VideoCapture(0)
while ok:
    ok, frame = capture.read()
    
    # binr = cv.threshold(frame, 0, 255, cv.THRESH_BINARY)[1] 
    # kernel = np.ones((5, 5), np.uint8) 
    # invert = cv.bitwise_not(binr) 
    # erosion = cv.erode(invert, kernel, iterations=1) 

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    limits = get_limits(colors)
    for color_name, hsv_value in limits.items():
        mask = cv.inRange(hsv, hsv_value[0], hsv_value[1])

        # mask = cv.GaussianBlur(mask, (15, 15), 0)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.erode(mask, kernel, iterations=3)
        # mask = cv.dilate(mask, kernel, iterations=3)

        cv.imshow(color_name, mask)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if w > 50 and h > 50:
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), colors[color_name], thickness=2)
                cv.putText(frame, color_name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[color_name], 2)
                print(f"{color_name} object at: x={x}, y={y}, width={w}, height={h}")

    cv.imshow("frame", frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()