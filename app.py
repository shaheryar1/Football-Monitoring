from Utils.Color import extract_color
import cv2
import time



if __name__ == '__main__':


    img = cv2.imread('test_images/red.jpg')
    H,W,C=img.shape

    img = img[0:int(H/2),:]


    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    now =time.time()
    ans=extract_color(img,visualize=True)
    print(time.time()-now)
    print(ans)