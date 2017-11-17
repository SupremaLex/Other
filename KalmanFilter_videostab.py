import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

vert_border = 5
horizontal_border = 5


def compareplots(oldtrajectory, newtrajectory, name):
    f = plt.figure(name)
    number = len(oldtrajectory)
    name = name.upper()
    if name == 'A':
        old = [math.atan2(e[1][0], e[0][0]) for e in oldtrajectory]
        new = [math.atan2(e[1][0], e[0][0]) for e in newtrajectory]
    else:
        if name == 'X':
            ind1 = 0
            ind2 = 2
        elif name == 'Y':
            ind1 = 1
            ind2 = 2
        old, new = [e[ind1][ind2] for e in oldtrajectory], [x[ind1][ind2] for x in newtrajectory]
    oldavg = np.average(old)
    newavg = np.average(new)
    # Среднеквадратическое отклонение
    s = sum([(e-oldavg)**2 for e in old])/(number-1)
    print('S = '+name, s**0.5)
    news = sum([(e-newavg)**2 for e in new])/(number-1)
    print('New S = '+name, news**0.5)
    frames = range(number)
    plt.plot(frames, old, figure=f, color='r')
    plt.plot(frames, new, figure=f, color='g')
    plt.xlabel('frame', figure=f)
    plt.ylabel(name, figure=f)


def videostab(filename):
    print('Video ' + filename + ' processed')
    X = np.array([0, 0, 0], dtype=np.float32)  # posteriori state estimate
    P = np.array([1, 1, 1], dtype=np.float32)  # posteriori estimate error covariance
    z = np.array([0, 0, 0], dtype=np.float32)  # actual measurement
    d = np.array([0, 0, 0], dtype=np.float32)  # change
    P_ = np.array([0, 0, 0], dtype=np.float32)
    X_ = np.array([0, 0, 0], dtype=np.float32)
    pstd = 5e-3  # can be changed
    cstd = 0.1  # can be changed
    Q = np.array([pstd, pstd, pstd])  # process noise covariance
    R = np.array([cstd, cstd, cstd])**2  # measurement noise covariance

    file = filename
    cap = cv2.VideoCapture(file)
    ret1, prev = cap.read()
    newsize = (1920//4, 1080//4)
    prev = cv2.resize(prev, newsize, cv2.INTER_CUBIC)
    prevgrey = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prevpts = cv2.goodFeaturesToTrack(prevgrey, **feature_params)

    old, new = [], []

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = cap.get(5)
    size1 = np.int(cap.get(3))
    size2 = np.int(cap.get(4))
    size = size1, size2
    videostab = file[:-4] + 'stab.mp4'
    out = cv2.VideoWriter(videostab, fourcc, fps, newsize)

    while cap.isOpened():

        # read frames
        ret2, cur = cap.read()
        if not ret2:
            break
        cur = cv2.resize(cur, newsize, cv2.INTER_CUBIC)

        affine = cv2.estimateRigidTransform(prev, cur, True)

        if not np.all(affine):
            affine = last_affine

        last_affine = affine
        old.append(affine)

        d[0] = affine[0][2]
        d[1] = affine[1][2]
        d[2] = math.atan2(affine[1][0], affine[0][0])
        z += d

        # prediction update
        P += Q  # P_(k) = P(k - 1) + Q
        # measurement update
        K = P_ / (P_ + R)     # gain K(k) = P_(k) / (P_(k) + R)
        X += K * (z - X)    # z - X_ is residual, X(k) = X_(k) + K(k) * (z(k) - X_(k))
        P *= (np.ones(3) - K)      # P(k) = (1 - K(k)) * P_(k)    delta update
        d += X - z

        cosptr = math.cos(d[2])
        sinptr = math.sin(d[2])
        newA = np.array([[cosptr, -sinptr, d[0]],
                         [sinptr, cosptr, d[1]]])

        cur2 = cv2.warpAffine(prev, newA, newsize)
        cur2 = cur2[vert_border: -vert_border][horizontal_border: -horizontal_border]
        cur2 = cv2.resize(cur2, newsize)
        new.append(newA)
        #cv2.imshow('original', prev)
        #cv2.imshow('stabilized', cur2)
        out.write(cur2)
        prev = cur
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    compareplots(oldtrajectory=old, newtrajectory=new, name='X')
    # y
    compareplots(oldtrajectory=old, newtrajectory=new, name='Y')
    # angle
    compareplots(oldtrajectory=old, newtrajectory=new, name='A')

    plt.show()
    return videostab


def rotate_video(filename, angle):
    cap = cv2.VideoCapture(filename)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = cap.get(5)
    size1 = np.int(cap.get(3))
    size2 = np.int(cap.get(4))
    size = size1, size2
    videostab = filename[-4]+'rotated.mp4'
    center = size1 // 2, size2 // 2
    rot = cv2.getRotationMatrix2D(center, angle, 1)
    out = cv2.VideoWriter(videostab, fourcc, fps, size)
    while cap.isOpened:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.warpAffine(frame, rot, size)
        out.write(frame)
    out.release()
    cap.release()


'''if __name__ == '__main__':
    name = videotab('VIDEO0008.mp4')'''

