import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


vert_border = 5
horizontal_border = 5


def compareplots(oldtrajectory, newtrajectory, name):
    """Draw plots of old image trajectory and image trajectory
    name in [A,X,Y]"""
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


def videostab(filename, pstd=4e-3, cstd=0.1, newsize=(640, 320)):
    """Simple video live stabilization via recursive Kalman Filter"""
    print('Video ' + filename + ' processed')
    # posteriori state estimate
    X = np.array([0, 0, 0], dtype=np.float32)
    # posteriori estimate error covariance
    P = np.array([1, 1, 1], dtype=np.float32)
    # actual measurement
    z = np.array([0, 0, 0], dtype=np.float32)
    # change
    d = np.array([0, 0, 0], dtype=np.float32)
    # process noise covariance
    Q = np.array([pstd, pstd, pstd])
    # measurement noise covariance
    R = np.array([cstd, cstd, cstd])**2

    file = filename
    cap = cv2.VideoCapture(file)
    nframes = np.int(cap.get(7))
    ret1, prev = cap.read()
    prev = cv2.resize(prev, newsize, cv2.INTER_CUBIC)

    old, new = [], []
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

        # save original affine for comparing with stabilizied
        old.append(affine)
        # get x, y and angle
        d[0] = affine[0][2]
        d[1] = affine[1][2]
        d[2] = math.atan2(affine[1][0], affine[0][0])
        # cumulative transform
        z += d
        # prediction update
        P += Q  # P(k) = P(k - 1) + Q
        # measurement update
        K = P / (P + R)     # gain K(k) = P(k) / (P(k) + R)
        X += K * (z - X)    # z - X is residual, X(k) = X_(k) + K(k) * (z(k) - X(k))
        P *= (np.ones(3) - K)      # P(k) = (1 - K(k)) * P(k)    delta update
        d += X - z
        # create new Affine transform
        cosptr = math.cos(d[2])
        sinptr = math.sin(d[2])
        newA = np.array([[cosptr, -sinptr, d[0]],
                         [sinptr, cosptr, d[1]]])

        # get stabilized frame
        cur2 = cv2.warpAffine(prev, newA, newsize)
        # crop borders
        cur2 = cur2[vert_border: -vert_border][horizontal_border: -horizontal_border]
        cur2 = cv2.resize(cur2, newsize)
        new.append(newA)
        # concatenate original and stabilized frames
        result = np.concatenate((cur2, cur))
        s1, s2 = result.shape[:2]
        # and rotate them
        rot = cv2.getRotationMatrix2D((s1//2, s2//2), 270, 1)
        result = cv2.warpAffine(result, rot, (s2, s1))
        cv2.imshow('show', result)
        out.write(cur2)
        prev = cur
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    # x
    compareplots(oldtrajectory=old, newtrajectory=new, name='X')
    # y
    compareplots(oldtrajectory=old, newtrajectory=new, name='Y')
    # angle
    compareplots(oldtrajectory=old, newtrajectory=new, name='A')

    plt.show()
    return videostab


if __name__ == '__main__':
    name = videostab('VIDEO0009.mp4', 0.007, 0.1)

