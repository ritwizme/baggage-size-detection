from scipy.spatial import distance as dist
from imutils import perspective
import imutils
import numpy as np
from imutils import contours
import argparse
import cv2

#finds the midpoint
def mp(pointA, pointB):
	return ((pointB[0] + pointB[0]) * 0.5, (pointB[1] + pointB[1]) * 0.5)

#parcing arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
ap.add_argument("-w", "--width", type=float, required=True)
args = vars(ap.parse_args())

#loading image
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

#finding contors
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

(cnts, _) = contours.sort_contours(cnts)
ppm = None

# loop over the contours individually
for c in cnts:
	if cv2.contourArea(c) < 100:
		continue

	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
	(tleft, tright, bright, bleft) = box
	(tltrX, tltrY) = mp(tleft, tright)
	(blbrX, blbrY) = mp(bleft, bright)

	(tlblX, tlblY) = mp(tleft, bleft)
	(trbrX, trbrY) = mp(tright, bright)


	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	if ppm is None:
		ppm = dB / args["width"]

	dimA = dA / ppm
	dimB = dB / ppm
	cv2.putText(orig,"Height:"+"{:.1f}in".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.55, (255, 255, 255), 1)
	cv2.putText(orig,"Width:"+"{:.1f}in".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.55, (255, 255, 255), 1)

	# show the output image
	cv2.imshow("Image", orig)
	cv2.waitKey(0)