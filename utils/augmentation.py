import random
import numpy as np
import scipy.ndimage
import torch
import torchvision.transforms.functional as tvf
import torch.nn.functional as tnf
import cv2
from PIL import Image
from imutils import paths


def hflip(image, labels):
    '''
    left-right flip

    Args:
        image: PIL.Image
        labels: tensor, shape(N,5), absolute x,y,w,h, angle in degree
    '''
    image = tvf.hflip(image)
    labels[:,0] = image.width - labels[:,0] # x,y,w,h,(angle)
    labels[:,4] = -labels[:,4]
    return image, labels


def vflip(image, labels):
    '''
    up-down flip

    Args:
        image: PIL.Image
        labels: tensor, shape(N,5), absolute x,y,w,h, angle in degree
    '''
    image = tvf.vflip(image)
    labels[:,1] = image.height - labels[:,1] # x,y,w,h,(angle)
    labels[:,4] = -labels[:,4]
    return image, labels


def rotate(image, degrees, labels, expand=False):
    '''
    image: PIL.Image
    labels: tensor, shape(N,5), absolute x,y,w,h, angle in degree
    '''
    img_w, img_h = image.width, image.height
    image = tvf.rotate(image, angle=-degrees, expand=expand)
    new_w, new_h = image.width, image.height
    # image coordinate to cartesian coordinate
    x = labels[:,0] - 0.5*img_w
    y = -(labels[:,1] - 0.5*img_h)
    # cartesian to polar
    r = (x.pow(2) + y.pow(2)).sqrt()

    theta = torch.empty_like(r)
    theta[x>=0] = torch.atan(y[x>=0]/x[x>=0])
    theta[x<0] = torch.atan(y[x<0]/x[x<0]) + np.pi
    theta[torch.isnan(theta)] = 0
    # modify theta
    theta -= (degrees*np.pi/180)
    # polar to cartesian
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    labels[:,0] = x + 0.5*new_w
    labels[:,1] = -y + 0.5*new_h
    labels[:,4] += degrees
    labels[:,4] = torch.remainder(labels[:,4], 180)
    labels[:,4][labels[:,4]>=90] -= 180

    return image, labels


def add_gaussian(imgs, max_var=0.1):
    '''
    imgs: tensor, (batch),C,H,W
    max_var: variance is uniformly ditributed between 0~max_var
    '''
    var = torch.rand(1) * max_var
    imgs = imgs + torch.randn_like(imgs) * var

    return imgs


def add_saltpepper(imgs, max_p=0.06):
    '''
    imgs: tensor, (batch),C,H,W
    p: probibility to add salt and pepper
    '''
    c,h,w = imgs.shape[-3:]

    p = torch.rand(1) * max_p
    total = int(c*h*w * p)

    idxC = torch.randint(0,c,size=(total,))
    idxH = torch.randint(0,h,size=(total,))
    idxW = torch.randint(0,w,size=(total,))
    value = torch.randint(0,2,size=(total,),dtype=torch.float)

    imgs[...,idxC,idxH,idxW] = value

    return imgs


def random_avg_filter(img):
    assert img.dim() == 3
    img = img.unsqueeze(0)
    ks = random.choice([3,5])
    pad_size = ks // 2
    img = tnf.avg_pool2d(img, kernel_size=ks, stride=1, padding=pad_size)
    return img.squeeze(0)


def max_filter(img):
    assert img.dim() == 3
    img = img.unsqueeze(0)
    img = tnf.max_pool2d(img, kernel_size=3, stride=1, padding=1)
    return img.squeeze(0)


def get_gaussian_kernels():
    gaussian_kernels = []
    for ks in [3,5]:
        delta = np.zeros((ks,ks))
        delta[ks//2,ks//2] = 1
        kernel = scipy.ndimage.gaussian_filter(delta, sigma=3)
        kernel = torch.from_numpy(kernel).float().view(1,1,ks,ks)
        gaussian_kernels.append(kernel)
    return gaussian_kernels

gaussian_kernels = get_gaussian_kernels()
def random_gaussian_filter(img):
    assert img.dim() == 3
    img = img.unsqueeze(1)
    kernel = random.choice(gaussian_kernels)
    assert torch.isclose(kernel.sum(), torch.Tensor([1]))
    pad_size = kernel.shape[2] // 2
    img = tnf.conv2d(img, weight=kernel, stride=1, padding=pad_size)
    return img.squeeze(1)


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(center,
                        scale,
                        rot,
                        output_size,
                        shift=np.array([0, 0], dtype=np.float32),
                        inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
            i *= 2
    return border // i


class AffineTransform:
    def __call__(self, img, labels, input_size, debug=True):
        output_size = input_size

        img = np.asarray(img)
        _h, _w = img.shape[:2]
        c = np.array([_w / 2., _h / 2.], dtype=np.float32)
        s = max(_w, _h) * 1.0
        s = s * np.random.choice(np.arange(0.8, 2., 0.1))
        w_border = _get_border(128, _w)
        h_border = _get_border(128, _h)
        c[0] = np.random.randint(low=w_border, high=_w - w_border)
        c[1] = np.random.randint(low=h_border, high=_h - h_border)
        r = 0 # np.random.randint(-180, 180) # TODO: affine with random rotate
    
        trans_input = get_affine_transform(
                        c, s, r, [input_size, input_size])
        img_tran = cv2.warpAffine(img, trans_input, 
                        (input_size, input_size),
                        flags=cv2.INTER_LINEAR)
        trans_output = get_affine_transform(c, s, r, [output_size, output_size])

        targets = torch.zeros(labels.shape[0], 5)

        for i, label in enumerate(labels):
            cx, cy, w, h, a = label
            x1, y1 = (cx - w//2, cy - h//2)
            x2, y2 = (cx + w//2, cy + h//2)
            x1y1 = affine_transform((x1, y1), trans_output)
            x2y2 = affine_transform((x2, y2), trans_output)
            x1_tran, y1_tran = x1y1[0], x1y1[1]
            x2_tran, y2_tran = x2y2[0], x2y2[1]

            w_tran, h_tran = x2_tran - x1_tran, y2_tran - y1_tran
            cx_tran = (x1_tran + x2_tran) // 2
            cy_tran = (y1_tran + y2_tran) // 2

            targets[i,:] = torch.tensor([cx_tran, cy_tran, w_tran, h_tran, a])

            if debug:
                # print(w_tran, h_tran)
                a_tran = a*np.pi/180
                C, S = np.cos(a_tran), np.sin(a_tran)
                R = np.asarray([[-C, -S], [S, -C]])
                pts = np.asarray([[-w_tran / 2, -h_tran / 2], [w_tran / 2, -h_tran / 2], 
                                    [w_tran / 2, h_tran / 2], [-w_tran / 2, h_tran / 2]])
                points = np.asarray([((cx_tran, cy_tran) + pt @ R).astype(int) for pt in pts])

                cv2.circle(img_tran, (int(cx_tran), int(cy_tran)), 2, (0, 255, 0), 5)
                cv2.polylines(img_tran, [points], True, (255, 0, 0), 2)

                # cv2.imwrite("img.jpg", img_tran)

        img_tran = Image.fromarray(img_tran)

        return img_tran, targets


if __name__ == "__main__":

    data_path = '/mnt/sdb1/Data/Fisheye'
    img_path = random.choice(list(paths.list_images(data_path)))
    print(img_path)

    label_path = img_path.replace("jpg", "txt")
    # img = cv2.imread(img_path)
    img = Image.open(img_path)
    
    # img.show("ori")

    affine_trans = AffineTransform()

    labels = []

    with open(label_path) as f:
        li = 0
        for i, line in enumerate(f):
            bbox = torch.tensor(list(map(float, line.rstrip("\n").split(" ")[1:])))
            labels.append(bbox)

    labels = torch.stack(labels)
    
    img_tran, labels_tran = affine_trans(img, labels)

    # img_tran.show("tran")
    cv2.imshow("f", img_tran)
    cv2.waitKey(0)