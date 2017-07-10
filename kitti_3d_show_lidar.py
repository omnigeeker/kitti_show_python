import os
import glob
import numpy as np
import cv2
import math
import mayavi.mlab as mlab

kitti_dir = "/ext/Data/Kitti/TrainingTest"
image_2_dir = os.path.join(kitti_dir, "training/image_2")
image_3_dir = os.path.join(kitti_dir, "training/image_3")
calib_dir = os.path.join(kitti_dir, "training/calib")
label_2_dir = os.path.join(kitti_dir, "training/label_2")
velodyne_dir = os.path.join(kitti_dir, "training/velodyne")

## preset view points
#  azimuth=180,elevation=0,distance=100,focalpoint=[0,0,0]
MM_TOP_VIEW  = 180, 0, 120, [0,0,0]
MM_PER_VIEW1 = 120, 30, 70, [0,0,0]
MM_PER_VIEW2 = 30, 45, 100, [0,0,0]
MM_PER_VIEW3 = 120, 30,100, [0,0,0]
## draw  --------------------------------------------

def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)

def imshow(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))

def make_color_with_far():
    pass

def draw_didi_lidar(fig, lidar, is_grid=1, is_axis=1):
    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]
    prs = np.clip(prs/15,0,1)

    #draw grid
    if is_grid:
        L=25
        dL=5
        Z=-2
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        for y in np.arange(-L,L+1,dL):
            x1,y1,z1 = -L, y, Z
            x2,y2,z2 =  L, y, Z
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.3,0.3,0.3), tube_radius=None, line_width=1, figure=fig)

        for x in np.arange(-L,L+1,dL):
            x1,y1,z1 = x,-L, Z
            x2,y2,z2 = x, L, Z
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.3,0.3,0.3), tube_radius=None, line_width=1, figure=fig)

    #draw axis
    if is_axis:
        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)

        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, line_width=2, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, line_width=2, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, line_width=2, figure=fig)

    mlab.points3d(
        pxs, pys, pzs, prs,
        mode='point',  # 'point'  'sphere'
        #colormap='spectral',  #(0.7,0.7,0.7),  #'gnuplot',  #'bone',  #'spectral',  #'copper',
        #color=(0.9,0.9,0.9),
        #color=(0.9,0.9,0),
        #scale_factor=1,
        figure=fig)
    # mlab.points3d(
    #     pxs, pys, pzs,
    #     mode='point',  # 'point'  'sphere'
    #     #colormap='bone',  #(0.7,0.7,0.7),  #'gnuplot',  #'bone',  #'spectral',  #'copper',
    #     #color=(0.9,0.9,0.9),
    #     #color=(0.9,0.9,0),
    #     scale_factor=1,
    #     figure=fig)

def draw_didi_boxes3d(fig, boxes3d, is_number=False, color=(1,1,1), line_width=1):

    if boxes3d.shape==(8,3): boxes3d=boxes3d.reshape(1,8,3)

    num = len(boxes3d)
    for n in range(num):
        b = boxes3d[n]

        if is_number:
            mlab.text3d(b[0,0], b[0,1], b[0,2], '%d'%n, scale=(1, 1, 1), color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

def mark_gt_box3d(mlab_fig,  velodyne_file, gt_boxes3d, size=(1000, 500), is_show=False):

    #os.makedirs(mark_dir, exist_ok=True)

    lidar = np.fromfile(velodyne_file, dtype=np.float32)
    lidar = lidar.reshape((-1, 4))
    #lidar   = np.load(velodyne_file)
    boxes3d = gt_boxes3d

    mlab.clf(mlab_fig)
    draw_didi_lidar(mlab_fig, lidar, is_grid=1, is_axis=1)
    if len(boxes3d)!=0:
        draw_didi_boxes3d(mlab_fig, boxes3d)

    #azimuth,elevation,distance,focalpoint = MM_PER_VIEW2
    #MM_PER_VIEW3 = 120, 30, 100, [0, 0, 0]
    azimuth, elevation, distance, focalpoint = -5, -75, 50, [20, 0, 0]
    mlab.view(azimuth,elevation,distance,focalpoint)
    mlab.savefig('1.png')
    if is_show:
        mlab.show()
    image = cv2.imread('1.png')
    cv2.putText(image, "LIDAR VIEW", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    return image

def get_boxes3d(label_2_file):
    gt_boxes3d = []
    for line in open(label_2_file):
        items = line.split(' ')
        type = items[0]
        truncated = items[1]
        occluded = items[2]
        alpha = items[3]
        h = np.float(float(items[8]))
        w = np.float(float(items[9]))
        l = np.float(float(items[10]))
        tx = np.float(float(items[11]))
        ty = np.float(float(items[12]))
        tz = np.float(float(items[13]))
        ry = np.float(float(items[14]))

        if type.lower() == 'dontcare':
            continue

        center = np.array([tx, ty, tz])
        R = np.array([[+math.cos(ry), 0, +math.sin(ry)],
                      [0, 1, 0],
                      [-math.sin(ry), 0, +math.cos(ry)]])

        P = np.array([[ 0.,  0.,  1.],
                      [-1.,  0.,  0.],
                      [ 0.,  -1., 0.]])
        # 3D bounding box corners
        x_corners = [l/2, l/2, -l/ 2, -l/ 2, l/ 2, l/ 2, -l/2, -l/2];
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2];

        #rotate and translate 3D bounding box
        corners_3D = np.dot(R, [x_corners, y_corners, z_corners])
        corners_3D_abs = (corners_3D.T + center).T
        corners_3D_abs = np.dot(P, corners_3D_abs)
        gt_boxes3d.append(corners_3D_abs.T)
    return np.array(gt_boxes3d)


def draw_3d_lidar_image(mlab_fig, velodyne_file, label_2_file):

    gt_boxes3d = get_boxes3d(label_2_file)
    mark_gt_box3d(mlab_fig, velodyne_file,gt_boxes3d, is_show=True)


if __name__ == '__main__':

    files = glob.glob(image_2_dir + "/*.png")
    training_count = len(files)

    mlab_fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None)

    for i in range(1, training_count):
        name = "{:06}".format(i)

        label_2_file = os.path.join(label_2_dir, name + ".txt")
        velodyne_file = os.path.join(velodyne_dir, name + ".bin")

        draw_image_2 = draw_3d_lidar_image(mlab_fig, velodyne_file, label_2_file)


