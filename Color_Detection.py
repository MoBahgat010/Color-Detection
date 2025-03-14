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
    'red': [139, 0, 0],
    'white': [255, 255, 255],
    'blue': [230, 216, 173],
    'green': [80, 215, 90],
    'yellow': [65, 210, 200],
    'orange': [80, 140, 235],
}
print(get_limits(colors))

ok = True
capture = cv.VideoCapture(0)
while ok:
    ok, frame = capture.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    limits = get_limits(colors)
    for color_name, hsv_value in limits.items():
        mask = cv.inRange(hsv, hsv_value[0], hsv_value[1])

        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.GaussianBlur(mask, (5, 5), 0)

        cv.imshow(color_name, mask)
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv.contourArea(contour)
            x, y, w, h = cv.boundingRect(contour)
            aspect_ratio = w / h
            if area > 1000 and 0.5 < aspect_ratio < 2.0:
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), color=colors[color_name], thickness=2)
                print(f"{color_name} object at: x={x}, y={y}, width={w}, height={h}")

    cv.imshow("frame", frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()