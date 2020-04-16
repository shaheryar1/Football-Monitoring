

import argparse
import os
import numpy as np
import cv2
# from Utils.Color_Extraction_RGB import extract_colors,hex_to_rgb
from Utils.ImageUtitls import *
from Utils.Color import extract_color
import dlib
import matplotlib.pyplot as plt
import io
import time
# import darknet as dn
from darknet.python import darknet as dn


def detection_image(image_path,net,meta,output_path,threshold=0.3):
    frame = cv2.imread(image_path)
    now = time.time()
    r = dn.detect(net, meta, image_path.encode('utf-8'), thresh=threshold)
    print(time.time()-now)

    for i, object in enumerate(r):
        # print(i,p)
        class_name = (str(object[0])[2:-1])
        # name = class_name + "-" + str(round(object[1] * 100)) + "%"
        score = str(round(object[1] * 100))
        rect = object[2]
        centerX, centerY, w, h = rect
        w = int(w)
        h = int(h)
        x1 = int(centerX - w / 2)
        y1 = int(centerY - h / 2)
        x2 = x1 + w
        y2 = y1 + h

        # rect = box.astype(int)
        # x1, y1, x2, y2 = rect
        box_color = (255, 190, 99)
        caption = class_name + " - " + score + "%"
        frame = drawBoxes(frame, (x1, y1, x2, y2), box_color, caption)
    img_name = str.split(image_path,'/')[-1]
    # print(img_name)

    cv2.imwrite(os.path.join(output_path,img_name),frame)


# def detect_and_track_video(video_path, net, meta,output_path, threshold=0.3):
#     #
#     # dal = VideoDAL()
#     #
#     # video_id = len(dal.getAllVideos()) + 1;
#     # name = str.split(video_path, '/')[-1]
#     # dal.insertVideo(video_id, name)
#     print(output_path)
#     cap = cv2.VideoCapture(video_path)
#     if (cap.isOpened() == False):
#         print("Error opening video stream or file")
#
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     print(frame_height)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(fps)
#     # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
#
#     i = 0;
#     start = time.time()
#     captions = [];
#     trackers = [];
#     t = dlib.correlation_tracker()
#     frame_rate_calc = 1
#     freq = cv2.getTickFrequency()
#
#     # Read until video is completed
#
#     print("Video fps :", fps)
#
#     while (cap.isOpened()):
#         ret, frame = cap.read()
#         t1 = cv2.getTickCount()
#         if(frame is None):
#             return
#         temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         if ret == True:
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             # Display the resulting frame
#
#             if (i % int((fps * 2)) == 0):
#                 captions = [];
#                 trackers = [];
#                 cv2.imwrite('a.jpg', frame)
#                 r = dn.detect(net, meta, 'a.jpg'.encode('utf-8'), thresh=threshold)
#                 boxes=[]
#                 scores=[]
#                 classes=[]
#                 for j, object in enumerate(r):
#                     # print(i,p)
#                     class_name = (str(object[0])[2:-1])
#                     # name = class_name + "-" + str(round(object[1] * 100)) + "%"
#                     score=str(round(object[1]*100))
#                     rect = object[2]
#                     centerX, centerY, w, h = rect
#                     w = int(w)
#                     h = int(h)
#                     x1 = int(centerX - w / 2)
#                     y1 = int(centerY - h / 2)
#                     x2=x1+w
#                     y2=y1+h
#                     boxes.append((x1, y1, x2, y2))
#                     classes.append(class_name)
#                     scores.append(score)
#
#                 print("before nms",len(boxes))
#                 nms_idx=non_max_suppression_fast(np.array(boxes),overlapThresh=0.5)
#                 print("after nms",len(nms_idx))
#                 for idx in nms_idx:
#                     box_color=(255, 190, 99)
#                     class_name=classes[idx]
#                     x1,y1,x2,y2=boxes[idx]
#                     score= scores[idx]
#                     caption = class_name + " - " + score + "%"
#                     frame = drawBoxes(frame, (x1, y1, x2, y2), box_color, caption)
#                     rect = dlib.rectangle(x1, y1, x2, y2)
#                     t = dlib.correlation_tracker()
#                     t.start_track(temp_frame, rect)
#                     # update our set of trackers and corresponding class
#                     captions.append(caption)
#                     trackers.append(t)
#
#                     # --- saving a snapshot -----
#                     # print("Taking snapshot at", i / int(fps))
#                     # bottom_roi = frame[int(y1 + (y2 - y1) / 2):y2, x1:x2]
#                     # bottom_roi = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2RGB)
#                     # # Extract colors
#                     # dominent_colors = list(extract_colors(bottom_roi, 2))
#                     # class_name = caption.split("-")[0]
#                     # score = caption.split("-")[1]
#                     # score = float(score[0:-1])
#                     # roi = frame[y1:y2, x1:x2]
#
#                     # take_snapshot(video_id, class_name, roi,
#                     #               confidence=score, dominent_colors=dominent_colors)
#
#
#
#             else:
#                 for (t, caption) in zip(trackers, captions):
#                     # update the tracker and grab the position of the tracked
#                     # object
#                     t.update(temp_frame)
#                     pos = t.get_position()
#                     # unpack the position object
#                     x1 = int(pos.left())
#                     y1 = int(pos.top())
#                     x2 = int(pos.right())
#                     y2 = int(pos.bottom())
#                     frame = drawBoxes(frame, (x1, y1, x2, y2), box_color, caption)
#
#
#         cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                     (255, 255, 0),
#                     2, cv2.LINE_AA)
#
#         t2 = cv2.getTickCount()
#         time1 = (t2 - t1) / freq
#         frame_rate_calc = 1 / time1
#         out.write(frame)
#         i = i + 1;
#         # frame=cv2.resize(frame,(0,0),fx=0.75,fy=0.75)
#         # cv2.imshow('Frame', frame)
#         if (i % int(fps) == 0):
#             print("Processed ", str(int(i / fps)), "seconds")
#
#         # Press Q on keyboard to  exit
#         # if cv2.waitKey(0) & 0xFF == ord('q'):
#         #     break
#         # else:
#         #     pass
#     # When everything done, release the video capture object
#     cap.release()
#     out.release()
#     # out.release()
#     # Closes all the frames
#     cv2.destroyAllWindows()




def detection_video(video_path, net, meta,output_path, threshold=0.3):

    print(output_path)
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(frame_height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

    i = 0;
    start = time.time()

    captions_track = []
    boxes_track = []
    colors_track=[]
    t = dlib.correlation_tracker()

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Read until video is completed
    print("Video fps :", fps)

    while (cap.isOpened()):
        ret, frame = cap.read()
        t1 = cv2.getTickCount()
        if(frame is None):
            return
        temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret == True:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display the resulting frame

            if (i % int((fps * 5)) == 0):
            # if(True):
                print("Detection")
                captions_track = []
                boxes_track = []
                colors_track = []
                cv2.imwrite('a.jpg', frame)
                r = dn.detect(net, meta, 'a.jpg'.encode('utf-8'), thresh=threshold)
                boxes=[]
                scores=[]
                classes=[]
                for j, object in enumerate(r):
                    # print(i,p)
                    class_name = (str(object[0])[2:-1])
                    # name = class_name + "-" + str(round(object[1] * 100)) + "%"
                    score=str(round(object[1]*100))
                    rect = object[2]
                    centerX, centerY, w, h = rect

                    try:

                        w = int(w)
                        h = int(h)
                        x1 = int(centerX - w / 2)
                        y1 = int(centerY - h / 2)
                        x2=x1+w
                        y2=y1+h


                        boxes.append((x1, y1, x2, y2))
                        classes.append(class_name)
                        scores.append(score)
                    except:
                        pass;

                # print("before nms",len(boxes))
                nms_idx=non_max_suppression_fast(np.array(boxes),overlapThresh=0.5)
                # nms_idx = np.arange(0,len(boxes))
                # print("after nms",len(nms_idx))

                n=1 #variable used for assigning tracker ids

                for idx in nms_idx:
                    box_color=(255, 190, 99)
                    class_name=classes[idx]
                    x1,y1,x2,y2=boxes[idx]
                    score= scores[idx]
                    caption = class_name + " - " + score + "%"

                    # Extracting color
                    roi=frame[y1:y2,x1:x2]

                    color_name = " ";
                    w,h,c=roi.shape
                    if(w==0 or h==0):
                        pass
                    else:
                        img = cv2.GaussianBlur(roi, (3, 3), cv2.BORDER_DEFAULT)
                        color_name,color_range=extract_color(img)
                        box_color=color_range[1]

                    rect = dlib.rectangle(x1, y1, x2, y2)
                    t = dlib.correlation_tracker()
                    t.start_track(temp_frame, rect)
                    # update our set of trackers and corresponding class
                    captions_track.append(n)
                    colors_track.append((color_name,box_color))
                    boxes_track.append(t)

                    frame = drawBoxes(frame, (x1, y1, x2, y2), box_color,str(n))
                    n=n+1
            else:
                for (t, caption,color) in zip(boxes_track, captions_track,colors_track):
                    # update the tracker and grab the position of the tracked
                    # object
                    t.update(temp_frame)
                    pos = t.get_position()
                    # unpack the position object
                    x1 = int(pos.left())
                    y1 = int(pos.top())
                    x2 = int(pos.right())
                    y2 = int(pos.bottom())
                    color_name,box_color=color
                    frame = drawBoxes(frame, (x1, y1, x2, y2), box_color, str(caption))


        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0),
                    2, cv2.LINE_AA)

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1
        out.write(frame)
        i = i + 1;
        frame=cv2.resize(frame,(0,0),fx=0.75,fy=0.75)
        cv2.imshow('Frame', frame)
        if (i % int(fps) == 0):
            print("Processed ", str(int(i / fps)), "seconds")
        # if(int(i/fps)>=2*60*fps):
        #     break

        # Press Q on keyboard to  exit
        # cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # else:
        #     pass
    # When everything done, release the video capture object
    cap.release()
    out.release()
    # out.release()
    # Closes all the frames
    cv2.destroyAllWindows()



model_cfg='darknet/cfg/yolov3.cfg'
model_weights='darknet/yolov3.weights'
meta_data='darknet/cfg/coco.data'


if __name__ == '__main__':

    net = dn.load_net(model_cfg.encode('utf-8'),
                      model_weights.encode('utf-8'), 0)
    meta = dn.load_meta(meta_data.encode('utf-8'))

    test_file = 'test.mp4'
    # for i in range(0,10):
    #     detection_image('test_images/red.jpqqqqqqqqqqqqqqqqg',net,meta,'results')
    detection_video(test_file, net, meta, output_path=os.path.join('results',test_file), threshold=0.5)

