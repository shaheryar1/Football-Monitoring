# Football-Monitoring
A project to monitor football games using stationary cam


### CLone the reporsitory using 
https://github.com/shaheryar1/Football-Monitoring.git

### Install dependencies using 

```pip3 install -r requirements.txt```


Open terminal and browse into Football-Monitoring repo

```cd Football-Monitoring```
### Clone and install darknet

```git clone https://github.com/pjreddie/darknet.git ```

```cd darknet```

```make```

To use CUDA follow original instructions here https://pjreddie.com/darknet/install/

### Download Yolov3 weights
``` wget https://pjreddie.com/media/files/yolov3.weights ```

Make sure you place these weights inside Football-Monitoring/darknet


### Run Detection  
run darknet_yolo_detector.py , by changing the "test_file" variable inside it
