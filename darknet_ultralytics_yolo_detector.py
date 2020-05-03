import os

# from Utils.Color_Extraction_RGB import extract_colors,hex_to_rgb
from Utils.ImageUtitls import *
from Utils.Color import extract_color
import dlib
import time
# import darknet as dn

from ultralytics import ultralytics as ul
from sort import Sort

def detection_video(video_path, net, meta, output_path, threshold=0.5, detection_refresh_rate=1, model_type=1):
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

    i = 0
    start = time.time()

    # To save last frame information globally
    main_captions_track = []
    main_boxes_track = []
    main_colors_track = []

    t = dlib.correlation_tracker()

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    model = None

    if model_type == 1:
        model, device = ul.load_ultralytics_model()

    # Read until video is completed
    print("Video fps :", fps)

    while (cap.isOpened()):
        ret, frame = cap.read()

        t1 = cv2.getTickCount()
        if (frame is None):
            return
        temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret == True:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display the resulting frame

            # if (i % int((fps * detection_refresh_rate)) == 0):
            if(True):
                print("Detection")
                captions_track = []
                boxes_track = []
                colors_track = []

                # print("saved frame")
                boxes = []
                scores = []
                classes = []


                num_preds = 0

                if model_type == 0:
                    r = dn.detect(net, meta, 'a.jpg'.encode('utf-8'), thresh=threshold)
                    for j, object in enumerate(r):
                        num_preds += 1
                        # j is the detection number and object is the returned predictions
                        # print(j, object)
                        class_name = (str(object[0])[2:-1])
                        # name = class_name + "-" + str(round(object[1] * 100)) + "%"
                        # Prediction score
                        score = str(round(object[1] * 100))
                        rect = object[2]
                        centerX, centerY, w, h = rect

                        try:

                            w = int(w)
                            h = int(h)
                            x1 = int(centerX - w / 2)
                            y1 = int(centerY - h / 2)
                            x2 = x1 + w
                            y2 = y1 + h

                            boxes.append((x1, y1, x2, y2))
                            classes.append(class_name)
                            scores.append(score)
                        except:
                            pass

                else:
                    boxes, classes, scores = ul.predict(frame, model, device)
                    num_preds = len(boxes)

                # print("before nms",len(boxes))
                nms_idx = non_max_suppression_fast(np.array(boxes), overlapThresh=0.5)
                # nms_idx = np.arange(0,len(boxes))
                # print("after nms",len(nms_idx))

                # List used for assigning tracker ids
                n = list(range(0, num_preds))

                for idx in nms_idx:
                    box_color = (255, 190, 99)
                    class_name = classes[idx]
                    x1, y1, x2, y2 = boxes[idx]
                    score = scores[idx]
                    caption = class_name + " - " + score + "%"

                    # Extracting color
                    roi = frame[y1:y2, x1:x2]
                    color_name = " "
                    w, h, c = roi.shape
                    if (w == 0 or h == 0):
                        pass
                    else:
                        img = cv2.GaussianBlur(roi, (3, 3), cv2.BORDER_DEFAULT)
                        color_name, color_range = extract_color(img)
                        box_color = color_range[1]

                    boxx = [x1, y1, x2, y2]
                    rect = dlib.rectangle(x1, y1, x2, y2)
                    t = dlib.correlation_tracker()
                    t.start_track(temp_frame, rect)

                    # update our set of trackers and corresponding class
                    # Getting all the Last Frame Data and matching IOUs
                    check = True
                    for p, (trc, caps, cols) in enumerate(
                            zip(main_boxes_track, main_captions_track, main_colors_track)):
                        pos = trc.get_position()
                        box = [int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())]

                        # If IOU better enough
                        if IoU(box, boxx) > 0.3:
                            col_name, bx_clr = cols

                            # To check if caption already assigned or not
                            # If assigned, pick the 1st one, else, use the one from previous tracker
                            if caps not in n:
                                caps = n.pop(0)
                            else:
                                n.pop(n.index(caps))

                            # Append the new tracker but with the new Caption
                            captions_track.append(caps)
                            colors_track.append(cols)
                            boxes_track.append(t)

                            frame = drawBoxes(frame,
                                              (x1, y1, x2, y2),
                                              bx_clr, str(caps))

                            # Removing assigned trackers and infos from list
                            main_boxes_track.pop(p)
                            main_captions_track.pop(p)
                            main_colors_track.pop(p)

                            check = False
                            break

                    if check:
                        # If IOU did not match, Pick last value from list and assign to the tracker
                        n_val = n.pop()
                        captions_track.append(n_val)
                        colors_track.append((color_name, box_color))
                        boxes_track.append(t)
                        frame = drawBoxes(frame, (x1, y1, x2, y2), box_color, str(n_val))


            else:
                for (t, caption, color) in zip(boxes_track, captions_track, colors_track):
                    # update the tracker and grab the position of the tracked
                    # object
                    t.update(temp_frame)
                    pos = t.get_position()
                    # unpack the position object
                    x1 = int(pos.left())
                    y1 = int(pos.top())
                    x2 = int(pos.right())
                    y2 = int(pos.bottom())

                    color_name, box_color = color
                    frame = drawBoxes(frame, (x1, y1, x2, y2), box_color, str(caption))

                # Saving last frame information for reuse after 5 secs
                main_captions_track = captions_track
                main_boxes_track = boxes_track
                main_colors_track = colors_track

        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0),
                    2, cv2.LINE_AA)

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1
        out.write(frame)
        i = i + 1
        frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
        cv2.imshow('Frame', frame)
        if (i % int(fps) == 0):
            print("Processed ", str(int(i / fps)), "seconds")


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


def detection_video_with_sort_tracker(video_path, output_path, threshold=0.5, detection_refresh_rate=1, model_type=1):
    print(output_path)
    mot_tracker = Sort(max_age=100)
    tracker_out = None
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
    i = 0
    start = time.time()


    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    model = None

    if model_type == 1:
        model, device = ul.load_ultralytics_model()

    # Read until video is completed
    print("Video fps :", fps)

    while (cap.isOpened()):
        ret, frame = cap.read()
        t1 = cv2.getTickCount()
        if (frame is None):
            return

        temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret == True:

            # if (i % int((fps * detection_refresh_rate)) == 0):
            if(True):
                boxes, classes, scores = ul.predict(frame, model, device)
                print("detections",len(boxes))
                # Apply NMS to detections
                nms_idx = non_max_suppression_fast(np.array(boxes), overlapThresh=0.5)
                # nms_idx = np.arange(0,len(boxes))

                tracker_in = []
                for idx in nms_idx:
                    box_color = (255, 190, 99)
                    class_name = classes[idx]
                    x1, y1, x2, y2 = boxes[idx]
                    score = scores[idx]
                    a = np.array([x1, y1, x2, y2,score])
                    a = a.astype(np.float)
                    tracker_in.append(a)

                    # caption = class_name + " - " + score + "%"
                #
                #     # Extracting color
                #     roi = frame[y1:y2, x1:x2]
                #     color_name = " "
                #     w, h, c = roi.shape
                #     # if (w == 0 or h == 0):
                #     #     pass
                #     # else:
                #     #     img = cv2.GaussianBlur(roi, (3, 3), cv2.BORDER_DEFAULT)
                #     #     color_name, color_range = extract_color(img)
                #     #     box_color = color_range[1]


                tracker_in = np.array(tracker_in)
                if (len(tracker_in) > 0):
                    tracker_out = mot_tracker.update(tracker_in)
                else:
                    tracker_out = mot_tracker.update()
            else:
                pass;

            # update tracker using tracker_in


            # Draw results
            if(tracker_out is not None and len(tracker_out)>0):
                for r in tracker_out:
                    r=r.astype(np.int)
                    # print(r)
                    x1,y1,x2,y2,id=r

                    frame = drawBoxes(frame, (x1, y1, x2, y2), box_color, str(id))

            # tracker_in=tracker_out





        # out.write(frame)
        i = i + 1
        frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)

        if (i % int(fps) == 0):


            print("Processed ", str(int(i / fps)), "seconds")
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1
        frame = cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 0),
                            2, cv2.LINE_AA)

        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
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





if __name__ == '__main__':
    test_file = 'WP vs FSCI.mp4'
    # for i in range(0,10):
    #     detection_image('test_images/red.jpg',net,meta,'results')
    detection_video_with_sort_tracker('test_videos/'+test_file, output_path=os.path.join('results', test_file), threshold=0.3,model_type=1)