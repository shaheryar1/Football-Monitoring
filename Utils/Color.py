import cv2
import numpy as np


# boudaries in BGR
COLORS = {
	"red":([17, 15, 100], [50, 56, 200]), #red
	"blue":([86, 31, 4], [220, 88, 50]), #blue
	"yellow":([25, 146, 190], [62, 174, 250]),#yellow

}

def count_nonblack_np(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.

    """
    return img.any(axis=-1).sum()
def extract_color(img,visualize=True):
	'''

	:param img: BGR image
	:param visualize: shows the resulting frame in a window
	:return: Name of the color
	'''


	max_ratio =0
	max_color=None

	for key in COLORS:
		lower,upper=COLORS[key]
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype="uint8")
		upper = np.array(upper, dtype="uint8")

		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(img, lower, upper)
		output = cv2.bitwise_and(img, img, mask=mask)

		tot_pix = count_nonblack_np(img)
		color_pix = count_nonblack_np(output)
		ratio = round(color_pix / tot_pix * 100,3)

		if(ratio>max_ratio):
			max_ratio=ratio
			max_color=key

		if(visualize):
			# show the images

			print(ratio)
			cv2.imshow("images", np.hstack([img, output]))
			cv2.waitKey(0)
	return max_color