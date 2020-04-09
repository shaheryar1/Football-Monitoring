from Utils.Color import extract_color
import cv2
import time

import sys

for p in sys.path:
    print(p)
if __name__ == '__main__':


    img = cv2.imread('test_images/yellow.jpg')
    H,W,C=img.shape

    img = img[0:int(H/2),:]

    now = time.time()
    img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

    ans=extract_color(img,visualize=True)
    print(time.time()-now)
    print(ans)