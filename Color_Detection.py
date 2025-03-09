import cv2 as cv
import numpy as np

def get_limits(color_dict):
    limits = {}
    for color_name, bgr_value in color_dict.items():
        c = np.uint8([[bgr_value]])
        hsvC = cv.cvtColor(c, cv.COLOR_BGR2HSV)

        lower_limit = hsvC[0][0][0] - 10, 100, 100
        upper_limit = hsvC[0][0][0] + 10, 255, 255

        lower_limit = np.array(lower_limit, dtype=np.uint8)
        upper_limit = np.array(upper_limit, dtype=np.uint8)

        limits[color_name] = (lower_limit, upper_limit)
    
    return limits

colors = {
    'red': [0, 0, 255],
    'blue': [255, 0, 0],
    'green': [0, 255, 0],
    'yellow': [0, 255, 255]
}

ok = True
capture = cv.VideoCapture(0)
while ok:
    ok, frame = capture.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    limits = get_limits(colors)
    for color_name, hsv_value in limits.items():
        mask = cv.inRange(hsv, hsv_value[0], hsv_value[1])
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if w > 15 and h > 15:
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), color=colors[color_name], thickness=2)
                print(f"{color_name} object at: x={x}, y={y}, width={w}, height={h}")

    cv.imshow("frame", frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()