import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7, 0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('calibration_wide/train (*).jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (7,7), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,7), corners, ret)
        write_name = 'corners_found/corners_found'+str(idx)+'.jpg'
        cv2.imwrite(write_name, img)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

import pickle

# Test undistortion on an image
img = cv2.imread('calibration_wide/test.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('res/test_undist.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "calibration_wide/wide_dist_pickle.p", "wb" ) )
# dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

# Visualize undistortion
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20,10))
ax1.imshow(img.astype(np.uint8))
ax1.set_title('Original Image', fontsize=10)
ax2.imshow(dst.astype(np.uint8))
ax2.set_title('Undistorted Image', fontsize=10)


# Affine Transformation
(h,w) = img.shape[0:2]
(cX,cY) = (w/2,h/2)
M = cv2.getRotationMatrix2D((cX,cY),45,1.0)
img_A = cv2.warpAffine(img,M,(w,h))
ax3.imshow(img_A.astype(np.uint8))
ax3.set_title('Affine Image', fontsize=10)
cv2.imwrite('res/test_Affine.jpg',img_A)

# Perspective Transformation
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]])
pts1 = np.float32([[0,0],[100,h-1],[200,h-1],[w-1,0]])
pts2 = np.float32([[0,0],[0,h-1],[300,300],[w-1,0]])

M1 = cv2.getPerspectiveTransform(pts,pts1)
M2 = cv2.getPerspectiveTransform(pts,pts2)

dst1 = cv2.warpPerspective(img,M1,(1080,1440))
dst2 = cv2.warpPerspective(img,M2,(1080,1440))
cv2.imwrite('res/test_per1.jpg',dst1)
cv2.imwrite('res/test_per2.jpg',dst2)
ax4.imshow(dst1.astype(np.uint8))
ax4.set_title('Perspective Image1', fontsize=10)
ax5.imshow(dst2.astype(np.uint8))
ax5.set_title('Perspective Image1', fontsize=10)

plt.show()