import copy
import os
import math
import torch
import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO, RTDETR
from Flow import Sort, KPTracker
from flowformeralg import Flowformeralg

C = (0, 540)
D = (1920,540)
confiedence = 0.4

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        default='./train/weights/best.pt')
    parser.add_argument('--image_resize', default=1280, type=int)
    parser.add_argument('--det_conf_thresh', default=0.4, type=float)
    parser.add_argument('--det_iou_thresh', default=0.6, type=float)
    parser.add_argument('--sort_max_age', default=60, type=int)
    parser.add_argument('--sort_min_hit', default=2, type=int)
    return parser.parse_args()

def xyxy2xywh(xmin, ymin, xmax, ymax):
    x = xmin
    y = ymin
    w = xmax-xmin
    h = ymax-ymin

    return x, y, w, h

def lines_intersect(A, B, C, D):
    # Extract coordinates of line AB
    x1, y1 = A
    x2, y2 = B

    # Extract coordinates of line CD
    x3, y3 = C
    x4, y4 = D

    # Calculate the direction of the lines
    direction_1 = (x4 - x3) * (y1 - y3) - (x1 - x3) * (y4 - y3)
    direction_2 = (x4 - x3) * (y2 - y3) - (x2 - x3) * (y4 - y3)
    direction_3 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    direction_4 = (x2 - x1) * (y4 - y1) - (x4 - x1) * (y2 - y1)

    # Check if the lines have crossed
    if (direction_1 * direction_2 < 0) and (direction_3 * direction_4 < 0):
        return True
    else:
        return False


class Detect:
    def __init__(self, weight, device, imgsz=1280):
        self.device = device
        self.imgsz = imgsz
        self.model = RTDETR(weight)
        self.model.info()

    def detectimg(self, img, conf, iou):
        results = self.model.predict(source=img, imgsz=1280, conf=conf, iou=iou)
        xyxy = results[0].boxes.xyxy.to('cpu').numpy()
        confi = results[0].boxes.conf.to('cpu').numpy().reshape(-1, 1)

        return np.concatenate((xyxy, confi), axis=1) 


if __name__ == "__main__":
    save_text = False
    imgpath = './vidoes/split'
    savepath = './results'
    os.makedirs(savepath, exist_ok = False)
    file = os.listdir(imgpath)
    data = [x for x in file if ".MP4" in x]
    savevideo = True

    for video in data:
        line_counting = 0
        pre_tracking = []
        video_id = video.split(".MP4")[0]
        cap = cv2.VideoCapture(os.path.join(imgpath, video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if savevideo == True:
            fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
            out1 = cv2.VideoWriter(f'{savepath}/{str(video_id)}.avi',
                            fourcc, fps, (1280, 720))
        initialcount = 0
        finalcount = 0
        bbox = []
        count = 0
        frame_id = 1

        result12 = []
        trackers1 = []
        args = parse_args()

        Detector = Detect(weight=args.weights, device='0', imgsz=args.image_resize)
        mot_tracker1 = Sort(args.sort_max_age, args.sort_min_hit)
        flowformer = Flowformeralg()
        colours = np.random.rand(32, 3) * 255
        ret, img11 = cap.read()
        result1 = Detector.detectimg(img11, args.det_conf_thresh, args.det_iou_thresh)
        det0 = result1[:, 0:5]
        for bbox in det0:
            x0, y0, w0, h0 = xyxy2xywh(bbox[0], bbox[1], bbox[2], bbox[3])
            if y0 <= C[1]:
                initialcount += 1
        pre_img11 = cv2.cvtColor(img11, cv2.COLOR_BGR2RGB)
        pre_img11 = torch.from_numpy(pre_img11).permute(2, 0, 1).float()

        all_pts1 = {}
        dep1 = {}
        while True:
            ret,img12 = cap.read()
            if ret == True:
                im12 = img12.copy()
                s = time.time()
                result12 = Detector.detectimg(img12, args.det_conf_thresh, args.det_iou_thresh)
                current_img12 = cv2.cvtColor(img12, cv2.COLOR_BGR2RGB)
                current_img12 = torch.from_numpy(current_img12).permute(2, 0, 1).float()
                flo1 = flowformer.compute_flow(pre_img11, current_img12)
                u1 = flo1[:, :, 0]
                v1 = flo1[:, :, 1]
                det1 = result12[:, 0:5]                        
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_frame >= total_frames:
                    for bbox in det1:
                        x0, y0, w0, h0 = xyxy2xywh(bbox[0], bbox[1], bbox[2], bbox[3])
                        if y0 >= C[1]:
                            finalcount += 1

                trackers1, trackerboxes1 = mot_tracker1.update(det1, flo1, 1)

                keep_line_idx1 = []
                if len(trackers1) != 0:

                    for p in trackers1:
                        depth = []
                        xmin = int(p[0])
                        ymin = int(p[1])
                        xmax = int(p[2])
                        ymax = int(p[3])
                        label = int(p[4])
                        keep_line_idx1.append(label)
                        if label in all_pts1:
                            all_pts1[label].append(((xmin + xmax) // 2, (ymin + ymax) // 2))
                        else:
                            all_pts1[label] = [((xmin + xmax) // 2, (ymin + ymax) // 2)]

                        cv2.putText(im12, 'T %d' % p[4], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.rectangle(im12, (xmin, ymin), (xmax, ymax), (
                            int(colours[label % 32, 0]), int(colours[label % 32, 1]), int(colours[label % 32, 2])), 3)
                        cv2.line(im12, C, D, (0,255,0), 3)

                        x, y, w, h = xyxy2xywh(xmin, ymin, xmax, ymax)

                # counting line
                if len(pre_tracking) != 0:
                    for id in pre_tracking:
                        xcp, ycp = pre_tracking[id][-1][0], pre_tracking[id][-1][1]
                        xc, yc = all_pts1[id][-1][0], all_pts1[id][-1][1]
                        A = (xcp, ycp)
                        B = (xc, yc)
                        cross = lines_intersect(A, B, C, D)
                        if cross == True:
                            line_counting += 1

                pre_tracking = copy.deepcopy(all_pts1)


                fps = 1. / float(time.time() - s)
                cv2.putText(im12, 'boll number: {} detection: {}'.format(line_counting, len(result12)),
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                if cv2.waitKey(1) & 0xff == 27:
                    break
                if savevideo == True:
                    im = cv2.resize(im12,(1280,720))
                    out1.write(im)
                pre_img11 = current_img12
            else:
                for bbox in det1:
                        x0, y0, w0, h0 = xyxy2xywh(bbox[0], bbox[1], bbox[2], bbox[3])
                        if y0 >= C[1]:
                            finalcount += 1
                KPTracker1.count = 0
                with open(f"{savepath}/counting_results.txt", "a+") as f:
                    context = "The number of video:{} under conf of :{} is :{}\n".format(video_id, args.det_conf_thresh, line_counting + initialcount + finalcount)
                    f.write(context)

                if savevideo == True:
                    out1.release()
                break
            frame_id += 1
    cv2.destroyAllWindows()





