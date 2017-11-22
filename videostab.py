import cv2
import numpy as np
from numpy.linalg import inv
import math
from matplotlib import pyplot as plt
import pickle
from kalmanfilter import KalmanFilter


vert_border = 10
horizontal_border = 10


def get_params_from_trajectory(trajectory):
    params = [[], [], [], [], [], []]
    for e in trajectory:
        params[0].append(e[0][0])  # a1
        params[1].append(e[0][1])  # a2
        params[2].append(e[0][2])  # b1
        params[3].append(e[1][0])  # a3
        params[4].append(e[1][1])  # a4
        params[5].append(e[1][2])  # b2
    return params


def trajectory(trajectory, color='r'):
    """Draw plots for all pairs matching params of old and new trajectory, such as old a1 vs new a2 etc."""
    number = len(trajectory)
    params = ('a1', 'a2', 'b1', 'a3', 'a4', 'b2')
    trajectories = dict(zip(params, get_params_from_trajectory(trajectory)))
    frames = range(number)
    for k in trajectories:
        f = plt.figure(k)
        plt.plot(frames, trajectories[k], figure=f, color=color)
        plt.xlabel('frame', figure=f)
        plt.ylabel(k, figure=f)


def videostab(filename, newsize=(640, 320)):
    """Simple video live stabilization via recursive Kalman Filter"""
    print('Video ' + filename + ' processed')
    with open('covariance.pickle', 'rb') as file:
        R = pickle.load(file)
    Q = np.diag([3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3])*0
    F = np.eye(6)
    X = np.zeros((6, 1))
    P = np.diag([1, 1, 1, 1, 1, 1])
    H = np.eye(6)
    kf_6 = KalmanFilter(X, F, H, P, Q, R)

    F = np.eye(3)
    X = np.zeros(3)
    P = np.ones(3)
    H = np.eye(3)
    Q = 4*np.ones(3)*1e-3
    R = np.ones(3)*0.1**2
    kf_3 = KalmanFilter(X, F, H, P, Q, R, 1)

    file = filename
    cap = cv2.VideoCapture(file)
    nframes = np.int(cap.get(7))
    ret1, prev = cap.read()
    prev = cv2.resize(prev, newsize, cv2.INTER_CUBIC)

    old, new_6, new_3 = [], [], []
    # videowriter args
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = cap.get(5)
    videostab = file[:-4] + 'stab.mp4'
    out = cv2.VideoWriter(videostab, fourcc, fps, newsize)

    for i in range(nframes-1):
        # read frames
        ret2, cur = cap.read()
        cur = cv2.resize(cur, newsize, cv2.INTER_CUBIC)
        affine = cv2.estimateRigidTransform(prev, cur, True)
        # Sometimes there is no Affine transform between frames, so we use the last
        if not np.all(affine):
            affine = last_affine
        last_affine = affine

        # save original affine for comparing with stabilized
        old.append(affine)
        kf_6.predict()
        z = np.array([affine.ravel()]).T
        kf_6.update(z)
        X = kf_6.x
        newtrans_6 = np.float32(np.array([[X[0], X[1], X[2]],
                                         [X[3], X[4], X[5]]]))
        # b1, b2, a
        d = affine[0][2], affine[1][2], math.atan2(affine[1][0], affine[0][0])
        kf_3.predict()
        kf_3.update(d)
        # create new Affine transform
        d = kf_3.d
        a11 = math.cos(d[2])
        a22 = math.sin(d[2])
        newtrans_3 = np.array([[a11, -a22, d[0]],
                              [a22, a11, d[1]]])
        # get stabilized frame
        cur2 = cv2.warpAffine(prev, newtrans_6, newsize)
        cur3 = cv2.warpAffine(prev, newtrans_3, newsize)
        # crop borders
        cur2 = cur2[vert_border: -vert_border][horizontal_border: -horizontal_border]
        cur2 = cv2.resize(cur2, newsize)
        cur3 = cur3[vert_border: -vert_border][horizontal_border: -horizontal_border]
        cur3 = cv2.resize(cur3, newsize)
        new_6.append(newtrans_6)
        new_3.append(newtrans_3)
        # concatenate original and stabilized frames
        result = concat_images(cur, cur2, cur3)
        cv2.imshow('show', result)
        out.write(cur2)
        prev = cur
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    trajectory(old, 'r')
    trajectory(new_6, 'g')
    trajectory(new_3, 'b')
    plt.show()
    return videostab


def concat_images(imga, imgb, imgc):
    """
    Combines 3 color image ndarrays side-by-side.
    """

    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    hc, wc = imgc.shape[:2]
    max_height = np.max([ha, hb, hc])
    total_width = wa+wb+wc
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.uint8)
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa+wb] = imgb
    new_img[:hc, wa+wb:wa+wb+wc] = imgc
    return new_img


def rotate_video(videoname):

    file = videoname
    cap = cv2.VideoCapture(file)
    nframes = np.int(cap.get(7))
    s1, s2 = np.int(cap.get(3)), np.int(cap.get(4))
    # videowriter args
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = cap.get(5)
    videostab = file[:-4] + 'rot.mp4'
    out = cv2.VideoWriter(videostab, fourcc, fps, (s2, s1))
    rot = cv2.getRotationMatrix2D((s1 // 2,  s2 // 2), -90, 1)

    for i in range(nframes-1):
        r, frame = cap.read()
        frame = cv2.warpAffine(frame, rot, (s2, s1))
        out.write(frame)
    cap.release()
    out.release()


if __name__ == '__main__':
    name = videostab('t1.mp4')

