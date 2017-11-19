import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


def matching(pathvideo, pathimage, **kwargs):
    """Find frames in video that similar to image
    i.e. image matching via SURF, SIFT or ORB"""
    start = time.time()
    detector = kwargs.get('detector') or 'SURF'
    minpercent = kwargs.get('minpercent') or 0.5
    size = kwargs.get('size') or (640, 480)
    show = kwargs.get('showmode') or False
    if show:
        matches_number_to_show = kwargs.get('nmatch') or 50
    if detector == 'SIFT':
        # create SIFT detector
        detector = cv2.xfeatures2d.SIFT_create(500)
        norm = cv2.NORM_L2
    if detector == 'ORB':
        # create ORB detector
        detector = cv2.ORB_create()
        norm = cv2.NORM_L2
    else:
        # create SURF detector
        detector = cv2.xfeatures2d.SURF_create(500)
        norm = cv2.NORM_L2
    # create Brute Force matcher
    bf = cv2.BFMatcher(norm, crossCheck=True)

    capture = cv2.VideoCapture(pathvideo)
    # get number of video frames
    nframes = np.int(capture.get(7))
    image = cv2.resize(cv2.imread(pathimage), size, cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # get image keypoints and descriptors
    image_kp, image_desc = detector.detectAndCompute(image, None)
    image_desc = np.float32(image_desc)

    output, coefs = [], []

    for i in range(nframes):
        # read frame
        ret, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, size, cv2.INTER_CUBIC)
        # get image keypoints and descriptors
        frame_kp, frame_desc = detector.detectAndCompute(frame, None)
        frame_desc = np.float32(frame_desc)
        # if there are no matches, bf.match raise System Error so we need try/except
        try:
            matches = bf.match(image_desc, frame_desc)
            matches = sorted(matches, key=lambda x: x.distance)
            if show:
                # NOT_DRAW_SINGLE_POINTS = 2
                args = {'singlePointColor': (255, 0, 0), 'flags': 2}
                n = min(len(matches), matches_number_to_show)
                out = cv2.drawMatches(image, image_kp, frame, frame_kp, matches[:n], None, **args)
                cv2.imshow('Matches', out)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            # Matching coefficient means the ratio of the number of matches to the average number of key points
            coefficient = 2*len(matches)/(len(frame_desc)+len(image_desc))
            print("Matching coefficient {:.2%} between image and frame number {:n}".format(coefficient, i))
            coefs.append(coefficient)
            # if coefficient is enough high add frame to output list
            if coefficient >= minpercent:
                output.append(frame)
        except SystemError:
            coefs.append(0)
            print("No matches")
    capture.release()
    print("ELAPSED TIME %.2f" % (time.time()-start))
    if not output:
        print('Minimal percent of similarity between image and frames is too high')
    # plot Coefficient vs frame number
    indmax = coefs.index(max(coefs))
    plt.plot(range(i), coefs[:i])
    plt.plot(range(i), [minpercent]*i, 'r')
    plt.annotate('max', xy=(indmax, max(coefs)), xytext=(indmax+15, max(coefs)+0.05),
                 arrowprops=dict(facecolor='black', arrowstyle="->", connectionstyle="arc3"),
                 )
    plt.xlabel('frame number')
    plt.ylabel('Matching coefficient')
    plt.show()
    return output


if __name__ == '__main__':
    params = {'detector': 'ORB', 'minpercent': 0.6, 'showmode': True, 'nmatch': 100}
    ftr = matching(pathvideo='t1.mp4', pathimage='tim4.jpg', **params)


