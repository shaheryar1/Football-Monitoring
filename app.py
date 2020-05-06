from Utils.Color import extract_color
import cv2
import time

import sys

if __name__ == '__main__':


    img = cv2.imread('test_images/a.jpg')
    H,W,C=img.shape


    # img = img[0:int(H/2),:]

    now = time.time()
    # cv2.resize(img,(0,0),fx=0.5,fy=0.5)
    # img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    print(time.time() - now)

    ans=extract_color(img,visualize=True)
    print(ans)
