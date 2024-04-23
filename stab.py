# 导入模块包
# from scipy import signal
# from scipy.ndimage import convolve
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import math
import torch
import matplotlib.pyplot as plt
# from pwcnet.PWCNet import Network as PWCNet
# from pwcnet.PWCNet import estimate as opticalFlowEstimate
import copy
import os
import cv2
import glob
import logging
import argparse
import numpy as np
from tqdm import tqdm
from alike.alike import ALike, configs
import time

# alike
class ImageLoader(object):
    def __init__(self, filepath: str):
        self.N = 3000
        if filepath.startswith('camera'):
            camera = int(filepath[6:])
            self.cap = cv2.VideoCapture(camera)
            if not self.cap.isOpened():
                raise IOError(f"Can't open camera {camera}!")
            logging.info(f'Opened camera {camera}')
            self.mode = 'camera'
        elif os.path.exists(filepath):
            if os.path.isfile(filepath):
                self.cap = cv2.VideoCapture(filepath)
                if not self.cap.isOpened():
                    raise IOError(f"Can't open video {filepath}!")
                rate = self.cap.get(cv2.CAP_PROP_FPS)
                self.N = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                duration = self.N / rate
                logging.info(f'Opened video {filepath}')
                logging.info(f'Frames: {self.N}, FPS: {rate}, Duration: {duration}s')
                self.mode = 'video'
            else:
                self.images = glob.glob(os.path.join(filepath, '*.png')) + \
                              glob.glob(os.path.join(filepath, '*.jpg')) + \
                              glob.glob(os.path.join(filepath, '*.ppm'))
                self.images.sort()
                self.N = len(self.images)
                logging.info(f'Loading {self.N} images')
                self.mode = 'images'
        else:
            raise IOError('Error filepath (camerax/path of images/path of videos): ', filepath)

    def __getitem__(self, item):
        if self.mode == 'camera' or self.mode == 'video':
            if item > self.N:
                return None
            ret, img = self.cap.read()
            if not ret:
                raise "Can't read image from camera"
            if self.mode == 'video':
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, item)
        elif self.mode == 'images':
            filename = self.images[item]
            img = cv2.imread(filename)
            if img is None:
                raise Exception('Error reading image %s' % filename)
        return img

    def __len__(self):
        return self.N

class SimpleTracker(object):
    def __init__(self):
        self.pts_prev = None
        self.desc_prev = None

    def update(self, img, pts, desc):
        N_matches = 0
        if self.pts_prev is None:
            self.pts_prev = pts
            self.desc_prev = desc

            out = copy.deepcopy(img)
            for pt1 in pts:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 1, (0, 0, 255), -1, lineType=16)
        else:
            matches = self.mnn_mather(self.desc_prev, desc)
            mpts1, mpts2 = self.pts_prev[matches[:, 0]], pts[matches[:, 1]]
            N_matches = len(matches)

            out = copy.deepcopy(img)
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, (0, 255, 0), lineType=16)
                cv2.circle(out, p2, 1, (0, 0, 255), -1, lineType=16)

            self.pts_prev = pts
            self.desc_prev = desc

        return out, N_matches

    def mnn_mather(self, desc1, desc2):
        sim = desc1 @ desc2.transpose()
        sim[sim < 0.9] = 0
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids1 = np.arange(0, sim.shape[0])
        mask = (ids1 == nn21[nn12])
        matches = np.stack([ids1[mask], nn12[mask]])
        return matches.transpose()
#定义SuperPoint网络
class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return semi, desc

#定义正向传播网络
class SuperPointFrontend(object):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
                 cuda=False):
        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh  # L2 descriptor distance for good match.
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.

        # Load the network in inference mode.
        self.net = SuperPointNet()
        if cuda:
            # Train on GPU, deploy on GPU.
            self.net.load_state_dict(torch.load(weights_path))
            self.net = self.net.cuda()
        else:
            # Train on GPU, deploy on CPU.
            self.net.load_state_dict(torch.load(weights_path,
                                                map_location=lambda storage, loc: storage))
        self.net.eval()

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, img):
        """ Process a numpy image to extract points and descriptors.
        Input
          img - HxW numpy float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
          """
        assert img.ndim == 2, 'Image must be grayscale.'
        assert img.dtype == np.float32, 'Image must be float32.'
        H, W = img.shape[0], img.shape[1]
        inp = img.copy()
        inp = (inp.reshape(1, H, W))
        inp = torch.from_numpy(inp)
        inp = torch.autograd.Variable(inp).view(1, 1, H, W)
        if self.cuda:
            inp = inp.cuda()
        # Forward pass of network.
        outs = self.net.forward(inp)
        semi, coarse_desc = outs[0], outs[1]
        semi= outs[0]
        # Convert pytorch -> numpy.
        semi = semi.data.cpu().numpy().squeeze()
        # --- Process points.
        dense = np.exp(semi)  # Softmax.
        dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            if self.cuda:
                samp_pts = samp_pts.cuda()
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc, heatmap

#特征点均匀化(SCS算法)  https://www.researchgate.net/publication/323388062_Efficient_adaptive_non-maximal_suppression_algorithms_for_homogeneous_spatial_keypoint_distribution
def ssc(keypoints, num_ret_points, tolerance, cols, rows):
    """
    特征点分布均匀化
    :param keypoints: 检测到的特征点
    :param num_ret_points:精确返回的特征点数目，default = 10
    :param tolerance:回点 数量的容差，default = 0.1
    :param cols:图像宽度 w
    :param rows:图像高度 h
    :return:
    """
    exp1 = rows + cols + 2 * num_ret_points
    exp2 = (
        4 * cols
        + 4 * num_ret_points
        + 4 * rows * num_ret_points
        + rows * rows
        + cols * cols
        - 2 * rows * cols
        + 4 * rows * cols * num_ret_points
    )
    exp3 = math.sqrt(exp2)
    exp4 = num_ret_points - 1

    sol1 = -round(float(exp1 + exp3) / exp4)  # first solution
    sol2 = -round(float(exp1 - exp3) / exp4)  # second solution

    high = (
        sol1 if (sol1 > sol2) else sol2
    )  # binary search range initialization with positive solution
    low = math.floor(math.sqrt(len(keypoints) / num_ret_points))

    prev_width = -1
    selected_keypoints = []
    result_list = []
    result = []
    complete = False
    k = num_ret_points
    k_min = round(k - (k * tolerance))
    k_max = round(k + (k * tolerance))

    while not complete:
        width = low + (high - low) / 2
        if (
            width == prev_width or low > high
        ):  # needed to reassure the same radius is not repeated again
            result_list = result  # return the keypoints from the previous iteration
            break

        c = width / 2  # initializing Grid
        num_cell_cols = int(math.floor(cols / c))
        num_cell_rows = int(math.floor(rows / c))
        covered_vec = [
            [False for _ in range(num_cell_cols + 1)] for _ in range(num_cell_rows + 1)
        ]
        result = []

        for i in range(len(keypoints)):
            row = int(
                math.floor(keypoints[i].pt[1] / c)
            )  # get position of the cell current point is located at
            col = int(math.floor(keypoints[i].pt[0] / c))
            if not covered_vec[row][col]:  # if the cell is not covered
                result.append(i)
                # get range which current radius is covering
                row_min = int(
                    (row - math.floor(width / c))
                    if ((row - math.floor(width / c)) >= 0)
                    else 0
                )
                row_max = int(
                    (row + math.floor(width / c))
                    if ((row + math.floor(width / c)) <= num_cell_rows)
                    else num_cell_rows
                )
                col_min = int(
                    (col - math.floor(width / c))
                    if ((col - math.floor(width / c)) >= 0)
                    else 0
                )
                col_max = int(
                    (col + math.floor(width / c))
                    if ((col + math.floor(width / c)) <= num_cell_cols)
                    else num_cell_cols
                )
                for row_to_cover in range(row_min, row_max + 1):
                    for col_to_cover in range(col_min, col_max + 1):
                        if not covered_vec[row_to_cover][col_to_cover]:
                            # cover cells within the square bounding box with width w
                            covered_vec[row_to_cover][col_to_cover] = True

        if k_min <= len(result) <= k_max:  # solution found
            result_list = result
            complete = True
        elif len(result) < k_min:
            high = width - 1  # update binary search range
        else:
            low = width + 1
        prev_width = width

    for i in range(len(result_list)):
        selected_keypoints.append(keypoints[result_list[i]])

    return selected_keypoints

#关键点转换成点函数
def keypointToPoint(keypoint):
    """
    keypoints 转换为 points
    :param keypoint: 关键点
    :return: 点
    """
    point = np.zeros(len(keypoint) * 2, np.float32)
    for i in range(len(keypoint)):
        point[i * 2] = keypoint[i].pt[0]
        point[i * 2 + 1] = keypoint[i].pt[1]
    point = point.reshape(-1, 2)
    return point

#裁剪视频函数
def fixBorder(frame):
    """
    边界裁剪
    :param frame:视频帧
    :return:裁剪后的视频帧
    """
    s = frame.shape
    # 在不移动中心的情况下，将图像缩放10%
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.05)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

# 列表读取、存储函数
def list_txt(path, list=None):
    '''
    存储列表为txt格式
    :param path: 储存list的位置
    :param list: list数据
    :return: None/relist 当仅有path参数输入时为读取模式将txt读取为list
             当path参数和list都有输入时为保存模式将list保存为txt
    '''
    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist

# 移动平均滤波器
def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # 定义过滤器
    f = np.ones(window_size) / window_size
    # 为边界添加填充
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # 应用卷积
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # 删除填充
    curve_smoothed = curve_smoothed[radius:-radius]
    # 返回平滑曲线
    return curve_smoothed

# 高斯低通滤波器
def gauss_ditong(trajectory, window, sigma):
  kernel = signal.gaussian(window, std=sigma)
  kernel = kernel / np.sum(kernel)
  return convolve(trajectory, kernel, mode='reflect')

# 自适应权重
def gauss(t, r, window_size):
  """
  @param: window_size ：平滑窗口的大小（r属于window_size）
  @param: t is 代表当前的时间点
  @param: r is 当前时间点的平滑半径

  Return:
          returns 影响时域平滑性的权重w（t,r）
  """
  return np.exp((-9 * (r - t) ** 2) / window_size ** 2)

# 自适应高斯平滑
def optimize_path(c, buffer_size=0, window_size=10):
    """
    @param: c is original camera trajectory
    @param: window_size is the hyper-parameter for the smoothness term


    Returns:
            returns an optimized gaussian smooth camera trajectory
    """
    lambda_t = 20
    if window_size > c.shape[0]:
        window_size = c.shape[0]

    P = Variable(c.shape[0])
    for t in range(c.shape[0]):
        # first term for optimised path to be close to camera path
        path_term = (P[t] - c[t]) ** 2

        # second term for smoothness using gaussian weights
        for r in range(window_size):
            if t - r < 0:
                break
            w = gauss(t, t - r, window_size)
            gauss_weight = w * (P[t] - P[t - r]) ** 2
            if r == 0:
                gauss_term = gauss_weight
            else:
                gauss_term += gauss_weight

        if t == 0:
            objective = path_term + lambda_t * gauss_term
        else:
            objective += path_term + lambda_t * gauss_term

    prob = Problem(Minimize(objective))
    prob.solve()
    return np.asarray(P.value)

# 自适应高斯平滑（实时）
def real_time_optimize_path(c, lambda_t, buffer_size=20, iterations=10, window_size=32, beta=1):
    """
    @param: c is camera trajectory within the buffer

    Returns:
        returns an realtime optimized smooth camera trajectory
    """

    # lambda_t = 100
    p = np.empty_like(c)

    W = np.zeros((buffer_size, buffer_size))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = gauss(i, j, window_size)

    bar = tqdm(total=c.shape[0] * c.shape[1])
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            y = [];
            d = None
            # real-time optimization
            for t in range(1, c.shape[2] + 1):
                if t < buffer_size + 1:
                    P = np.asarray(c[i, j, :t])
                    if not d is None:
                        for _ in range(iterations):
                            alpha = c[i, j, :t] + lambda_t * np.dot(W[:t, :t], P)
                            alpha[:-1] = alpha[:-1] + beta * d
                            gamma = 1 + lambda_t * np.dot(W[:t, :t], np.ones((t,)))
                            gamma[:-1] = gamma[:-1] + beta
                            P = np.divide(alpha, gamma)
                else:
                    P = np.asarray(c[i, j, t - buffer_size:t])
                    for _ in range(iterations):
                        alpha = c[i, j, t - buffer_size:t] + lambda_t * np.dot(W, P)
                        alpha[:-1] = alpha[:-1] + beta * d[1:]
                        gamma = 1 + lambda_t * np.dot(W, np.ones((buffer_size,)))
                        gamma[:-1] = gamma[:-1] + beta
                        P = np.divide(alpha, gamma)
                d = np.asarray(P);
                y.append(P[-1])
            p[i, j, :] = np.asarray(y)
            bar.update(1)
    bar.close()
    return p

# def smooth(trajectory,adaptive_weights):
def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # 过滤x, y和角度曲线
    for i in range(3):
        # 移动平均平滑
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=40)
        # 高斯滤波器
        # smoothed_trajectory[:, i] = gauss_ditong(trajectory[:, i], 1000, 30)
        # 高斯自适应(不用)
        # smoothed_trajectory[:,i] = optimize_path(trajectory[:,i])
        # 高斯自适应实时
        # smoothed_trajectory[:,i] = real_time_optimize_path(np.asarray(np.expand_dims(np.expand_dims(trajectory[:,i], axis=0), axis=0)),adaptive_weights)
        # print(smoothed_trajectory[:,i])

    return smoothed_trajectory
# 读取输入视频



parser = argparse.ArgumentParser()
parser.add_argument('--InputBasePath', default='ceshi.mp4')
#parser.add_argument('--cuda', action='store_true', help='use GPU computation')
opt = parser.parse_args()
print(opt)
# 离线视频读取
cap = cv2.VideoCapture(opt.InputBasePath)

# 实时摄像头读取
# cap = cv2.VideoCapture(0)

# 获取视频流的宽度和高度
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(n_frames)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 获取每秒帧数(fps)
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)
#定义输出视频编码器

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#稳像前后对比视频
out_compare = cv2.VideoWriter('out_compare.mp4', fourcc, 25, (2*w, h))
#稳像后视频便于指标计算
out_video = cv2.VideoWriter('out.mp4', fourcc, 25, (w,h))
# outcut_video = cv2.VideoWriter('zooming_cut.avi', fourcc, 25, (w,h))
# 预定义转换numpy矩阵
transforms = np.zeros((n_frames, 3), np.float32)
tlist = []
prev_gray = []
prev = []
k = 0

#lk光流法参数
lk_params = dict(winSize  = (15, 15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

# 定义卡尔曼滤波参数
X = np.zeros((1,3))
P = np.ones((1,3))
R = np.array([0.25,0.25,0.25])
Q = np.array([4e-3,4e-3,4e-3])

# 载入关键点检测网络参数
# print('==>加载SuperPoint网络.')
# tt = time.perf_counter()
# # This class runs the SuperPoint network and processes its outputs.
# fe = SuperPointFrontend(weights_path="superpoint_v1.pth",
#                         nms_dist=4,
#                         conf_thresh=0.015,
#                         nn_thresh=0.7,
#                         cuda=True)
# tt_ = time.perf_counter()
# print('==> 加载成功.')
# print('耗时：'+str(tt_-tt))

print('==>加载ALIKE网络.')
ta = time.perf_counter()
model = ALike(**configs['alike-t'],
              device='cuda',
              top_k=-1,
              scores_th=0.2,
              n_limit=5000)
ta_ = time.perf_counter()
print('==> 加载成功.')
print('耗时：'+str(ta_-ta))

# # 加载PWC光流网络
# print('==>加载PWCnet网络.')
# tp = time.perf_counter()
# # Build model
# pwc = PWCNet()
# pwc.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load('pwc.pytorch').items()})
# # pwc = PWCNet.load_state_dict('network-default.pytorch')
# pwc = pwc.cuda()
# pwc.eval()
# tp_ = time.perf_counter()
# print('==> 加载成功.')
# print('耗时：'+str(tp_-tp))

# 加载liteflownet光流网络
# print('==>加载liteflownet网络.')
# tl = time.perf_counter()
# # Build model
# liteflownet = Network()
# liteflownet.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
#                              torch.load('liteflownet/liteflownet.pytorch').items()})
# liteflownet = liteflownet.cuda()
# tl_ = time.perf_counter()
# print('==> 加载成功.')
# print('耗时：'+str(tl_-tl))

#高斯自适应权重存储列表
adaptive_weights = []
# 绘制平滑轨迹
x_path = []
smooth_xpath = []
y_path = []
smooth_ypath = []
a_path = []
smooth_apath = []

while cap.isOpened():
    if k == 200:
        break
    # 读取一帧
    success, curr = cap.read()
    # 是否还有下一帧，关闭
    if not success:
        break

    # 转换为灰度图
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # 为了计算帧差，要把前几帧放入列表中
    prev.append(curr)
    prev_gray.append(curr_gray)

    if len(prev_gray) >= 2:
        # 检测前一帧的特征点

        # # 用shi-tomasi角点
        # start = time.clock()
        # prev_pts = cv2.goodFeaturesToTrack(prev_gray,maxCorners=500,qualityLevel=0.01,minDistance=30,blockSize=3)

        # # 用ORB特征点
        # start = time.clock()
        # keypoints1, descriptors1 = cv2.ORB_create().detectAndCompute(prev_gray[k-1], None)
        # prev_pts = keypointToPoint(keypoints1)

        # # 用surf特征点
        # start = time.clock()
        # keypoints1, descriptors1 = cv2.xfeatures2d_SURF.create().detectAndCompute(prev_gray[k-1], None)
        # prev_pts = keypointToPoint(keypoints1)

        # # 用SIFT特征点
        # start = time.clock()
        # keypoints1, descriptors1 = cv2.xfeatures2d_SIFT.create().detectAndCompute(prev_gray[k-1], None)
        # prev_pts = keypointToPoint(keypoints1)

        # 用superPoint特征点
        # start1 = time.perf_counter()
        # image1 = prev_gray[k-1].astype(np.float32) / 255.
        # keypoints1, descriptors1, h1 = fe.run(image1)
        # keypoints1 = [cv2.KeyPoint(keypoints1[0][i], keypoints1[1][i], 1) for i in range(keypoints1.shape[1])]

        # 用ALIKE关键点
        start1 = time.perf_counter()
        image1 = prev[k-1]
        pred = model(image1, sub_pixel=not False)

        keypoints1 = pred['' \
                          'keypoints']
        keypoints1 = np.array(keypoints1).reshape(-1, 1, 2)
        keypoints1 = [cv2.KeyPoint(keypoints1[i][0][0], keypoints1[i][0][1], 1) for i in range(keypoints1.shape[0])]
        t_zipo = time.perf_counter()
        print(t_zipo-start1)

        # 对特征点进行均匀化处理
        keypoints1 = sorted(keypoints1, key=lambda x: x.response, reverse=True)
        keypoints1 = ssc(keypoints1, 256, 0.1, w, h)#其他
        # 将特征点转化为可跟踪的角点
        prev_pts = keypointToPoint(keypoints1)

        # 计算光流(即轨迹特征点) 前一张 当前张 前一张特征

        # 用稠密光流
        # flow = cv2.calcOpticalFlowFarneback(prev_gray[k - 1], curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #
        # # 用pwcnet
        # t_flow1 = time.time()
        #
        # tensorInputFirst = (torch.from_numpy((prev[k - 1]/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)).cuda()
        # tensorInputSecond = (torch.from_numpy((curr/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)).cuda()
        # flow = opticalFlowEstimate(tensorInputFirst,tensorInputSecond,pwc)
        # flow = flow.data.cpu()
        # flow = flow[0].numpy().transpose((1, 2, 0))
        #
        # t_flow2 = time.time()
        # print(t_flow2-t_flow1)

        # 对关键点坐标进行位移
        # for i, point in enumerate(keypoints1):
        #     x, y = point.pt
        #     dx, dy = flow[int(y), int(x)]
        #     keypoints1[i].pt = (x + dx, y + dy)
        # curr_pts = keypointToPoint(keypoints1)
        # t_flow3 = time.time()
        # print(t_flow3-t_flow2)
        # # 用LK光流
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray[k - 1], curr_gray, prev_pts, None, **lk_params)

        # 检查完整性
        assert prev_pts.shape == curr_pts.shape

        # 只过滤有效点
        # idx = np.where(status == 1)[0]
        # prev_pts = prev_pts[idx]
        # curr_pts = curr_pts[idx]

        # # 找到变换矩阵
        m, inlier = cv2.estimateAffine2D(prev_pts, curr_pts)

        # 提取traslation
        dx = m[0, 2]
        dy = m[1, 2]

        # 提取旋转角
        da = np.arctan2(m[1, 0], m[0, 0])

        # homography, _ = cv2.findHomography(np.array(prev_pts), np.array(curr_pts), cv2.RANSAC)
        #
        # # 计算高斯自适应平滑权重(根据MeshFlow原文进行)
        # sorted_eigenvalue_magnitudes = np.sort(np.abs(np.linalg.eigvals(homography)))
        #
        # translational_element = math.sqrt((homography[0, 2] / w) ** 2 + (homography[1, 2] / h) ** 2)
        # affine_component = sorted_eigenvalue_magnitudes[-2] / sorted_eigenvalue_magnitudes[-1]
        #
        # adaptive_weight_candidate_1 = -1.93 * translational_element + 0.95
        # #在实际运行过程中发现要更改(+)为(-),根据这篇文章：https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization/issues/12#issuecomment-553737073
        # adaptive_weight_candidate_2 = 5.83 * affine_component + 4.88
        # if (adaptive_weight_candidate_1 > adaptive_weight_candidate_2):
        #     adaptive_weights_candidate = adaptive_weight_candidate_2
        # else:
        #     adaptive_weights_candidate = adaptive_weight_candidate_1
        # if adaptive_weights_candidate < 0:
        #     adaptive_weights_candidate = 0
        #     pass
        # adaptive_weights.append(adaptive_weights_candidate)

        # 存储转换
        transforms[k] = [dx, dy, da]

        end1 = time.perf_counter()
    k += 1

    if len(prev_gray) >= 2:
        start2 = time.perf_counter()
        # # 使用累积变换和计算轨迹
        trajectory = np.cumsum(transforms, axis=0)
        x_path.append(trajectory[k][0])
        y_path.append(trajectory[k][1])
        a_path.append(trajectory[k][2])

        # # 卡尔曼滤波开始
        if k == 1:
            #初始化
            X = np.array([0,0,0])
            P = np.array([1,1,1])
        else:
            #预测
            X_ = X
            P_ = P + Q
            #更新
            K = P_/(P_ + R)
            X = X_ +K * (trajectory - X_)
            P = (np.ones((1,3)) - K) * P_
        smoothed_trajectory = X
        # smoothed_trajectory = trajectory
        # 创建变量来存储平滑的轨迹
        # smoothed_trajectory = smooth(smoothed_trajectory, adaptive_weights_candidate)
        smoothed_trajectory = smooth(smoothed_trajectory)

        smooth_xpath.append(smoothed_trajectory[k][0])
        smooth_ypath.append(smoothed_trajectory[k][1])
        smooth_apath.append(smoothed_trajectory[k][2])
        # 计算smoothed_trajectory与trajectory的差值
        difference = smoothed_trajectory - trajectory

        # 计算更新的转换数组
        transforms_smooth = transforms + difference

        # 从新的转换数组中提取转换
        dx = transforms_smooth[k, 0]
        dy = transforms_smooth[k, 1]
        da = transforms_smooth[k, 2]

        # 根据新的值重构变换矩阵
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        VERTICAL_BORDER = 60
        HORIZONTAL_BORDER = 80

        # 应用仿射包装到给定的框架
        frame_stab = cv2.warpAffine(curr, m, (w, h))
        end2 = time.perf_counter()
        # out_video.write(frame_stab)
        # 按一定比例裁剪视频
        frame_stab = frame_stab[VERTICAL_BORDER:-VERTICAL_BORDER, HORIZONTAL_BORDER:-HORIZONTAL_BORDER]

        # frame_stab = fixBorder(frame_stab)
        frame_stab = cv2.resize(frame_stab, (w, h), interpolation=cv2.INTER_CUBIC)
        out_video.write(frame_stab)
        # 保存稳定视频

        # outcut_video.write(frame_stab)

        tlist.append((end1-start1)+(end2-start2))
        print("Frame: " + str(k-1) + "/" +
              " -  Tracked points : " + str(len(prev_pts)) +
              " -  time : " + str((end1-start1)+(end2-start2)))
        # 展示稳像前后对比视频
        frame_compare = cv2.hconcat([curr, frame_stab])
        # 保存对比视频
        out_compare.write(frame_compare)
        if (frame_compare.shape[1] > 1280):
            frame_compare = cv2.resize(frame_compare, (int(frame_compare.shape[1] / 2), int(frame_compare.shape[0] / 2)),interpolation=cv2.INTER_CUBIC)

        # 保存时间
        # list_txt('time.txt',tlist)

        cv2.imshow("Before and After", frame_compare)

        c = cv2.waitKey(10)
        if c == 27:
            break

plt.plot(np.asarray(x_path))
plt.plot(np.asarray(smooth_xpath))
plt.savefig('stable-xpath.png')
plt.clf()
plt.plot(np.asarray(y_path))
plt.plot(np.asarray(smooth_ypath))
plt.savefig('stable-ypath.png')
plt.clf()
plt.plot(np.asarray(a_path))
plt.plot(np.asarray(smooth_apath))
plt.savefig('stable-apath.png')
plt.clf()


