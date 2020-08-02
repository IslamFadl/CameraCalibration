import numpy as np
import cv2
import glob
import pickle

def main():

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((12*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:12, 0:9].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('Images/cam0*.bmp')

    # Step through the list and search for chessboard corners
    i = 0
    for idx, fname in enumerate(images):
        i += 1
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (12,9), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (12,9), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.bmp'
            #cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(50)
            print(i)
    cv2.destroyAllWindows()


    # Test undistortion on an image
    # Example for image path: 'Images/cam0_10.bmp'
    image_path = input('Please enter an image path: ')
    img = cv2.imread(image_path, 0)
    img_size = (img.shape[1], img.shape[0])


    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    print(dist)
    # save unistorted image.
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('Images/test_undist.bmp', dst)

    # Save the camera calibration result for later use in a pickle file
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("Images/dist_pickle.p", "wb"))

"""
 I found it difficult to visualize the distortion for the given data set.
 Nevertheless, it is visualized with the code provided in the
 jupyter notebook.
"""

if __name__ == '__main__':
    main()