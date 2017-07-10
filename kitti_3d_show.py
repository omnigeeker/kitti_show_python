import glob
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cv2
from kitti_3d_show_lidar import *

header_row = ['type', 'truncation', 'oocclusin', 'alpha', \
                  'x1', 'y1', 'x2', 'y2', 'h', 'w', 'l',
                  'tx', 'ty', 'tz', 'r']

def totuple(x):
    return (int(x[0]),int(x[1]))

def drawBox2D(draw_img, corners_2d, color, thickness=2) :
    # x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2,  l / 2, -l / 2, -l / 2];
    # y_corners = [    0,     0,      0,      0,    -h,     -h,     -h,     -h];
    # z_corners = [w / 2,-w / 2, -w / 2,  w / 2, w / 2, -w / 2, -w / 2,  w / 2];
    assert (len(corners_2d) == 8)

    #x,y
    cv2.line(draw_img, totuple(corners_2d[0]), totuple(corners_2d[1]), color, thickness)
    cv2.line(draw_img, totuple(corners_2d[2]), totuple(corners_2d[3]), color, thickness)
    cv2.line(draw_img, totuple(corners_2d[4]), totuple(corners_2d[5]), color, thickness)
    cv2.line(draw_img, totuple(corners_2d[6]), totuple(corners_2d[7]), color, thickness)
    #x,z
    cv2.line(draw_img, totuple(corners_2d[0]), totuple(corners_2d[4]), color, thickness)
    cv2.line(draw_img, totuple(corners_2d[1]), totuple(corners_2d[5]), color, thickness)
    cv2.line(draw_img, totuple(corners_2d[2]), totuple(corners_2d[6]), color, thickness)
    cv2.line(draw_img, totuple(corners_2d[3]), totuple(corners_2d[7]), color, thickness)
    #y,z
    cv2.line(draw_img, totuple(corners_2d[0]), totuple(corners_2d[3]), color, thickness)
    cv2.line(draw_img, totuple(corners_2d[1]), totuple(corners_2d[2]), color, thickness)
    cv2.line(draw_img, totuple(corners_2d[4]), totuple(corners_2d[7]), color, thickness)
    cv2.line(draw_img, totuple(corners_2d[5]), totuple(corners_2d[6]), color, thickness)

def drawline2D(draw_img, line_2d, color, thicknes=2):
    assert (len(line_2d) == 2)
    cv2.line(draw_img, totuple(line_2d[0]), totuple(line_2d[1]), color, thicknes)

def make_3dlabel_to_image(image, label_file, P, R_rect, image_name="LEFT CEMARE"):
    image2 = np.copy(image)
    for line in open(label_file):
        items = line.split(' ')
        type = items[0]
        truncated = items[1]
        occluded = items[2]
        alpha = items[3]
        x1 = np.int(float(items[4]))
        y1 = np.int(float(items[5]))
        x2 = np.int(float(items[6]))
        y2 = np.int(float(items[7]))
        h = np.float(float(items[8]))
        w = np.float(float(items[9]))
        l = np.float(float(items[10]))
        tx = np.float(float(items[11]))
        ty = np.float(float(items[12]))
        tz = np.float(float(items[13]))
        ry = np.float(float(items[14]))

        if type.lower() == 'dontcare':
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,255), 1)
            continue

        #assert(len(items) == 15)
        color = (255,255,255)
        if type.lower() == 'car': color =(0,255,0)
        elif type.lower() == 'van': color = (0,0,255)
        elif type.lower() == 'pedestrian': color = (255,0,0)
        elif type.lower() == 'cyclist': color = (255,255,0)
        elif type.lower() == 'truck': color = (0, 0, 255)

        # Draw2D
        #cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        #cv2.putText(image, type, (x1,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

        # Draw3D
        center = np.array([tx, ty, tz])
        # compute rotational matrix around yaw axis
        R = np.array([[+math.cos(ry), 0, +math.sin(ry)],
                      [0, 1, 0],
                      [-math.sin(ry), 0, +math.cos(ry)]])

        # 3D bounding box corners
        x_corners = [l/2, l/2, -l/ 2, -l/ 2, l/ 2, l/ 2, -l/2, -l/2];
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2];

        #rotate and translate 3D bounding box
        corners_3D = np.dot(R, [x_corners, y_corners, z_corners])
        corners_3D_abs = (corners_3D.T + center).T
        if any([s < 0.1 for s in corners_3D_abs[2]]): continue

        ones = [[1. for i in range(8)]]
        corners_3D_abs = np.concatenate((corners_3D_abs, ones), axis=0)

        #D3_to_D2
        #corners_2D_ = np.dot(np.dot(P, R_rect), corners_3D_abs)
        corners_2D_ = np.dot(P, corners_3D_abs)
        corners_2D_ = np.divide(corners_2D_,corners_2D_[2])
        corners_2D = corners_2D_[0:2]

        drawBox2D(image2, corners_2D.T, color, 1)
        cv2.putText(image2, type, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        #draw orientation_3D
        orientation_3D = np.array([[0.,l],[0.,0.],[0.,0.]])
        orientation_3D = np.dot(R, orientation_3D);
        orientation_3D_abs = (orientation_3D.T + center).T
        if any([s < 0.1 for s in orientation_3D_abs[2]]): continue

        orientation_3D_abs = np.concatenate((orientation_3D_abs, [[1.,1.]]), axis=0)
        orientation_2D_ = np.dot(P, orientation_3D_abs)
        orientation_2D_ = np.divide(orientation_2D_, orientation_2D_[2])
        orientation_2D = orientation_2D_[0:2]
        drawline2D(image2, orientation_2D.T, (0,255,255), 2)

    cv2.putText(image2, image_name, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

    #return np.concatenate((image, image2), axis=0)
    return image2

# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================
def point_cloud_2_top(points,
                      res=0.1,
                      zres=0.3,
                      side_range=(-10., 10.),  # left-most to right-most
                      fwd_range=(-10., 10.),  # back-most to forward-most
                      height_range=(-2., 2.),  # bottom-most to upper-most
                      ):
    """ Creates an birds eye view representation of the point cloud data for MV3D.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        zres:        (float)
                    Desired resolution on Z-axis in metres to use.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        numpy array encoding height features , density and intensity.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    reflectance = points[:,3]

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    z_max = int((height_range[1] - height_range[0]) / zres)
    # z_max =
    top = np.zeros([y_max+1, x_max+1, z_max+1], dtype=np.float32)

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and(
        (x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and(
        (y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)


    # # ASSIGN EACH POINT TO A HEIGHT SLICE
    # # n_slices-1 is used because values above max_height get assigned to an
    # # extra index when we call np.digitize().
    # bins = np.linspace(height_range[0], height_range[1], num=n_slices-1)
    # slice_indices = np.digitize(z_points, bins=bins, right=False)
    # # RESCALE THE REFLECTANCE VALUES - to be between the range 0-255
    # pixel_values = scale_to_255(r_points, min=0.0, max=1.0)
    # FILL PIXEL VALUES IN IMAGE ARRAY
    # -y is used because images start from top left
    # x_max = int((side_range[1] - side_range[0]) / res)
    # y_max = int((fwd_range[1] - fwd_range[0]) / res)
    # im = np.zeros([y_max, x_max, n_slices], dtype=np.uint8)
    # im[-y_img, x_img, slice_indices] = pixel_values

    for i, height in enumerate(np.arange(height_range[0], height_range[1], zres)):

        z_filt = np.logical_and((z_points >= height),
                                (z_points < height + zres))
        zfilter = np.logical_and(filter, z_filt)
        indices = np.argwhere(zfilter).flatten()

        # KEEPERS
        xi_points = x_points[indices]
        yi_points = y_points[indices]
        zi_points = z_points[indices]
        ref_i = reflectance[indices]

        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        x_img = (-yi_points / res).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (-xi_points / res).astype(np.int32)  # y axis is -x in LIDAR

        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor & ceil used to prevent anything being rounded to below 0 after
        # shift
        x_img -= int(np.floor(side_range[0] / res))
        y_img += int(np.floor(fwd_range[1] / res))

        # CLIP HEIGHT VALUES - to between min and max heights
        pixel_values = zi_points - height_range[0]
        # pixel_values = zi_points

        # FILL PIXEL VALUES IN IMAGE ARRAY
        top[y_img, x_img, i] = pixel_values

        # max_intensity = np.max(prs[idx])
        top[y_img, x_img, z_max] = ref_i

    return top

def draw_top_image(top):
    top_binary = np.zeros_like(top)
    top_binary[top > 0] = 128
    return np.dstack((top_binary, top_binary, top_binary)).astype(np.uint8)

def point_cloud_2_front(points,
                   res = 0.1,
                   z_res = 0.1,
                   side_range=(-30., 30.),  # left-most to right-most
                   height_range=(-2., 0.4),  # bottom-most to upper-most
                   ):
    # EXTRACT THE POINTS FOR EACH AXIS
    points = points[points[:,0]>0]
    points = points[points[:,1]>side_range[0]]
    points = points[points[:,1]<side_range[1]]
    points = points[points[:,2]>height_range[0]]
    points = points[points[:,2]<height_range[1]]

    y_division = res
    z_division = z_res
    quantized = (points - [0, side_range[0], height_range[0], 0]) / [1, y_division,z_division, 1]

    Y0, Yn = 0, int((side_range[1]-side_range[0])//y_division)+1
    Z0, Zn = 0, int((height_range[1]- height_range[0])//z_division)+1
    height  = Zn - Z0
    width   = Yn - Y0
    grid = [[[] for y in range(width)] for z in range(height)]
    front = np.zeros(shape=(height, width), dtype=np.float32)

    print("height:{} - width:{}".format(height, width))

    for i in range(len(quantized)):
        grid[int(quantized[i][2])][int(quantized[i][1])].append(quantized[i][0])

    for z in range(height):
        for y in range(width):
            if len(grid[z][y]) > 0:
                #print("z:{} - y:{}".format(z, y))
                front[height-z-1][width-y-1] = min(grid[z][y])
    return front

def draw_front_image(front):
    image = np.zeros_like(np.dstack((front, front, front)).astype(np.uint8))
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            z = front[y][x]
            if z == 0.:
                image[y][x] = [0, 0, 0]
            elif z <= 50:
                alpha = float(z - 0) / 50.
                image[y][x] = [0, int(255 * alpha), int(255 * (1 - alpha))]     # B,G,R
            elif z <= 100:
                 alpha = float(z - 100) / 100.
                 image[y][x] = [int(255 * alpha), int(255 * (1 - alpha)), 0]    # B,G,R
            else:
                image[y][x] = [255, 0, 0]   # B,G,R
    return image



def make_3dlabel_to_lidar_topview(velodyne_file, label_file, t_velo_to_cam, thickness=2):
    scan = np.fromfile(velodyne_file, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    side_range = (-30., 30.)
    fwd_range = (-30., 60.)
    height_range = (-2., 1.)
    res = 0.1

    top_view = point_cloud_2_top(scan, res=res, zres=0.3,
                                  side_range=side_range,  # left-most to right-most
                                  fwd_range=fwd_range,  # back-most to forward-most
                                  height_range=height_range)
    image = draw_top_image(top_view[:, :, -1])

    for line in open(label_file):
        items = line.split(' ')
        type = items[0]
        truncated = items[1]
        occluded = items[2]
        alpha = items[3]
        x1 = np.int(float(items[4]))
        y1 = np.int(float(items[5]))
        x2 = np.int(float(items[6]))
        y2 = np.int(float(items[7]))
        h = np.float(float(items[8]))
        w = np.float(float(items[9]))
        l = np.float(float(items[10]))
        tx = np.float(float(items[11]))
        ty = np.float(float(items[12]))
        tz = np.float(float(items[13]))
        ry = np.float(float(items[14]))

        if type.lower() == 'dontcare':
            continue

        # assert(len(items) == 15)
        color = (255, 255, 255)
        if type.lower() == 'car':
            color = (0, 255, 0)
        elif type.lower() == 'van':
            color = (0, 0, 255)
        elif type.lower() == 'pedestrian':
            color = (255, 0, 0)
        elif type.lower() == 'cyclist':
            color = (255, 255, 0)
        elif type.lower() == 'truck':
            color = (0, 0, 255)

        # Draw3D
        center = np.array([tx, ty, tz])
        # compute rotational matrix around yaw axis
        R = np.array([[+math.cos(ry), 0, +math.sin(ry)],
                      [0, 1, 0],
                      [-math.sin(ry), 0, +math.cos(ry)]])

        # 3D bounding box corners
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

        # rotate and translate 3D bounding box
        corners_3D = np.dot(R, [x_corners, y_corners, z_corners])
        corners_3D_abs = (corners_3D.T + center).T

        corners_top = corners_3D_abs.T[:4, [0,2]]

        center_2d = np.array([side_range[1]/res, fwd_range[1]/res])
        corners_image = center_2d - (corners_top*[-1,1] - t_velo_to_cam[[0,2]])/res
        cv2.line(image, totuple(corners_image[0]), totuple(corners_image[1]), color, thickness)
        cv2.line(image, totuple(corners_image[1]), totuple(corners_image[2]), color, thickness)
        cv2.line(image, totuple(corners_image[2]), totuple(corners_image[3]), color, thickness)
        cv2.line(image, totuple(corners_image[3]), totuple(corners_image[0]), color, thickness)
    cv2.putText(image, "BIRD VIEW", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    return image

def make_3dlabel_to_lidar_frontview(velodyne_file, label_file, t_velo_to_cam):
    scan = np.fromfile(velodyne_file, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    side_range = (-30., 30.)
    height_range = (-3., 5.)
    res = 0.05
    z_res = 0.05

    front_view = point_cloud_2_front(scan, res=res, z_res=z_res,
                            side_range=side_range, height_range=height_range)

    image = draw_front_image(front_view)
    cv2.putText(image, "FRONT VIEW", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    return image

def load_calib(calib_file):
    P0, P1, P2, P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo, t_velo_to_cam, t_imu_to_velo = None,None,None,None,None,None,None,None,None
    for line in open(calib_file):
        items = line.split(' ')
        if items[0] == "P0:":
            array = np.array([float(items[i]) for i in range(1,13)])
            P0 = array.reshape(3,4)
        elif items[0] == "P1:":
            array = np.array([float(items[i]) for i in range(1,13)])
            P1 = array.reshape(3,4)
        elif items[0] == "P2:":
            array = np.array([float(items[i]) for i in range(1,13)])
            P2 = array.reshape(3,4)
        elif items[0] == "P3:":
            array = np.array([float(items[i]) for i in range(1,13)])
            P3 = array.reshape(3,4)
        elif items[0] == "R0_rect:":
            array = np.array([float(items[i]) for i in range(1, 10)])
            R0_rect = array.reshape(3,3)
            R0_rect = np.concatenate((R0_rect, [[0.,0.,0.]]), axis=0)
            R0_rect = np.concatenate((R0_rect, [[0.], [0.], [0.], [1.]]), axis=1)
        elif items[0] == "Tr_velo_to_cam:":
            array = np.array([float(items[i]) for i in range(1, 13)])
            Tr_velo_to_cam = array.reshape(3, 4)
            Tr_velo_to_cam = np.concatenate((Tr_velo_to_cam, [[0.,0., 0.,1.]]), axis=0)
            t_velo_to_cam = Tr_velo_to_cam.T[3][:3]
        elif items[0] == "Tr_imu_to_velo:":
            array = np.array([float(items[i]) for i in range(1, 13)])
            Tr_imu_to_velo = array.reshape(3, 4)
            Tr_imu_to_velo = np.concatenate((Tr_imu_to_velo, [[0.,0.,0.,1.]]), axis=0)
            t_imu_to_velo = Tr_imu_to_velo.T[3][:3]
    return P0, P1, P2, P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo, t_velo_to_cam, t_imu_to_velo

if __name__ == '__main__':

    kitti_dir = "/ext/Data/Kitti/TrainingTest"
    image_2_dir = os.path.join(kitti_dir, "training/image_2")
    image_3_dir = os.path.join(kitti_dir, "training/image_3")
    calib_dir = os.path.join(kitti_dir, "training/calib")
    label_2_dir = os.path.join(kitti_dir, "training/label_2")
    velodyne_dir = os.path.join(kitti_dir, "training/velodyne")
    output_dir = "output"

    files = glob.glob(image_2_dir+"/*.png")
    training_count = len(files)
    image_demo = cv2.imread(files[0])
    print(image_demo.shape)
    mlab_fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(image_demo.shape[1], image_demo.shape[0]))

    for i in range(1,training_count):
        name = "{:06}".format(i)

        image_2_file = os.path.join(image_2_dir, name + ".png")
        image_3_file = os.path.join(image_3_dir, name + ".png")
        calib_file = os.path.join(calib_dir, name + ".txt")
        label_2_file = os.path.join(label_2_dir, name + ".txt")
        velodyne_file = os.path.join(velodyne_dir, name + ".bin")
        output_file = os.path.join(output_dir, name + ".png")

        P0, P1, P2, P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo, t_velo_to_cam, t_imu_to_velo = load_calib(calib_file)

        draw_image_2 = make_3dlabel_to_image(cv2.imread(image_2_file), label_2_file, P2, R0_rect, "LEFT CEMARE")
        draw_image_2 = cv2.resize(draw_image_2, (int(draw_image_2.shape[1] * 300 / draw_image_2.shape[0]), 300))
        draw_image_3 = make_3dlabel_to_image(cv2.imread(image_3_file), label_2_file, P3, R0_rect, "RIGHT CEMARE")
        draw_image_3 = cv2.resize(draw_image_3, (int(draw_image_3.shape[1] * 300 / draw_image_3.shape[0]), 300))

        gt_boxes3d = get_boxes3d(label_2_file)
        lidar_image = mark_gt_box3d(mlab_fig, velodyne_file, gt_boxes3d)
        lidar_image = cv2.resize(lidar_image, (draw_image_2.shape[1], draw_image_2.shape[0]))

        draw_lidar_front_view = make_3dlabel_to_lidar_frontview(velodyne_file, label_2_file, t_velo_to_cam)
        draw_lidar_front_view = cv2.resize(draw_lidar_front_view, (draw_image_2.shape[1], int(draw_lidar_front_view.shape[0]*draw_image_2.shape[1]/draw_lidar_front_view.shape[1])))
        draw_image = np.concatenate((draw_lidar_front_view, lidar_image, draw_image_2, draw_image_3), axis=0)

        draw_lidar_top_view = make_3dlabel_to_lidar_topview(velodyne_file, label_2_file, t_velo_to_cam)
        draw_lidar_top_view = cv2.resize(draw_lidar_top_view, (int(draw_lidar_top_view.shape[1]*draw_image.shape[0]/draw_lidar_top_view.shape[0]), draw_image.shape[0]))
        draw_image = np.concatenate((draw_lidar_top_view, draw_image), axis=1)

        if 0:
            wait_time = 10000
            cv2.imshow("img", draw_image)
            cv2.waitKey(int(wait_time))
        cv2.imwrite(output_file, draw_image)
        print("write {}".format(output_file))
