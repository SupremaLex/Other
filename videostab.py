import cv2
import numpy as np
from numpy.linalg import inv
import math
from matplotlib import pyplot as plt
import pickle
from kalmanfilter import KalmanFilter


vert_border = 10
horizontal_border = 10
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))


def curves(prevcorners, prev, cur):
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

    #corners = cv2.goodFeaturesToTrack(prev, mask=None, **feature_params)
    mask = np.zeros_like(cur)

    nextcorners, status, err = cv2.calcOpticalFlowPyrLK(prev, cur, prevcorners, None, **lk_params)

    good_new = nextcorners[status == 1]
    good_old = prevcorners[status == 1]
    tracks = []
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        y = np.int32(new.ravel())
        x = np.int32(old.ravel())
        tracks.append((y, x))
        mask = cv2.line(cur, (y[0],y[1]), (x[0], x[1]), (0, 0, 0), 2)

    img = cv2.add(cur, mask)
    return good_new, img, tracks


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
    Q = np.diag([1e-6, 1e-6, 4e-3, 1e-6, 1e-6, 5e-1])
    F = np.eye(6)
    X = np.zeros((6, 1))
    P = np.diag([1, 1, 1, 1, 1, 1])
    H = np.eye(6)
    kf_6 = KalmanFilter(X, F, H, P, Q, R)

    F = np.eye(3)
    X = np.zeros(3)
    P = np.ones(3)
    H = np.eye(3)
    Q = np.array([4e-3, 5e-3, 1e-4])
    R = np.array([4e-4, 4e-4, 1e-5])*100
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

        # Threshold for an optimal value, it may vary depending on the image.
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

        cur2grey = cv2.cvtColor(cur2, cv2.COLOR_BGR2GRAY)
        cur3grey = cv2.cvtColor(cur3, cv2.COLOR_BGR2GRAY)
        img = 0


        new_6.append(newtrans_6)
        new_3.append(newtrans_3)
        # concatenate original and stabilized frames
        if np.all(img): cur2 = img
        result = concat_images(cur, cur2, cur3)
        cv2.imshow('show', result)
        out.write(cur2)
        prev = cur
        prev2 = cur2grey
        prev3 = cur3grey
        if i > 1:
            good_old2 = good_new2[:]
            good_old3 = good_new3[:]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    trajectory(old, 'r')
    #trajectory(new_6, 'g')
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

def show(videoname):
    cap = cv2.VideoCapture(videoname)
    size = 640, 480
    nframes = np.int(cap.get(7))
    ret, prev = cap.read()
    prev = cv2.resize(prev, size)
    prevgrey = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(prevgrey, 50, 0.01, 5)
    new_tracks = []
    for i in range(nframes-2):
        ret, cur = cap.read()
        cur = cv2.resize(cur, size)
        #cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        nextcorners, img, tracks = curves(corners, prev, cur)
        pts = np.array([tracks])
        new_tracks.append(pts)
        #pts = tracks.reshape((-1, 1, 2))
        print([e[0][:1] for e in new_tracks])
        #if i % 10 == 0:
        cur = cv2.polylines(cur, np.array([e[0][:1] for e in new_tracks]), False, (0, 0, 0), thickness=1, lineType=8)
        cv2.imshow('show', cur)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    #name = videostab('t1.mp4', (480, 480))
    show('t1.mp4')

