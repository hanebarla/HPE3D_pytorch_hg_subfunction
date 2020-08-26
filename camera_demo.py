from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import cv2
import numpy as np
import torch
import torch.utils.data
from opts import opts
from model import create_model
from utils.debugger import Debugger
from utils.image import get_affine_transform, transform_preds
from utils.eval import get_preds, get_preds_3d

image_ext = ['jpg', 'jpeg', 'png']
mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)


class dcam(Debugger):
    def __init__(self):
        self.loop_on = 1
        super().__init__()

    def realtime_show(self, pause=False, k=0):
        max_range = np.array([self.xmax - self.xmin, self.ymax - self.ymin, self.zmax - self.zmin]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (self.xmax + self.xmin)
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (self.ymax + self.ymin)
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (self.zmax + self.zmin)
        for xb, yb, zb in zip(Xb, Yb, Zb):
            self.ax.plot([xb], [yb], [zb], 'w')
        self.plt.draw()
        self.plt.pause(0.1)
        self.plt.cla()

    def press(self, event):
        if event.key == 'escape':
            self.loop_on = 0

    def destroy_loop(self):
        self.fig.canvas.mpl_connect('key_press_event', self.press)


def is_image(file_name):
    ext = file_name[file_name.rfind('.') + 1:].lower()
    return ext in image_ext


def demo_image(image, model, opt):
    s = max(image.shape[0], image.shape[1]) * 1.0
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    trans_input = get_affine_transform(
        c, s, 0, [opt.input_w, opt.input_h])
    inp = cv2.warpAffine(image, trans_input, (opt.input_w, opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp / 255. - mean) / std
    inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    inp = torch.from_numpy(inp).to(opt.device)
    out = model(inp)[-1]
    pred = get_preds(out['hm'].detach().cpu().numpy())[0]
    pred = transform_preds(pred, c, s, (opt.output_w, opt.output_h))
    pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(),
                           out['depth'].detach().cpu().numpy())[0]

    return image, pred, pred_3d


def main(opt):
    camera = cv2.VideoCapture(0)
    opt.heads['depth'] = opt.num_output
    if opt.load_model == '':
        opt.load_model = '../models/fusion_3d_var.pth'
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
    else:
        opt.device = torch.device('cpu')

    model, _, _ = create_model(opt)
    model = model.to(opt.device)
    model.eval()

    debugger = dcam()
    k = 0

    while debugger.loop_on:
        ret, frame = camera.read()
        image, pred, pred_3d = demo_image(frame, model, opt)

        debugger.add_img(image)
        debugger.add_point_2d(pred, (255, 0, 0))
        debugger.add_point_3d(pred_3d, 'b')
        debugger.realtime_show(k)
        debugger.destroy_loop()
        debugger.show_all_imgs()

        k = cv2.waitKey(10)
        if k == 27:
            debugger.loop_on = 0

    cv2.destroyAllWindows()
    camera.release()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
