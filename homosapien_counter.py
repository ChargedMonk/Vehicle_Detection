import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.3,
    'gpu': 1.0
}

tfnet = TFNet(options)

vidcap = cv2.VideoCapture(0)
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("sec.jpg", image)     # save frame as JPG file
    return hasFrames
sec = 0
counter = 1
frameRate = 0.5
#it will capture image in each 0.5 second
success = getFrame(sec)
while success:
    sec = sec + frameRate
    sec = round(sec, 2)
    img = cv2.imread('sec.jpg', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # use YOLO to predict the image
    result = tfnet.return_predict(img)

    numberofpersons = 0
    labels = [li['label'] for li in result]
    for i in labels:
        if i == 'person':
            numberofpersons = numberofpersons + 1

    print('No. of Persons =', numberofpersons)
    counter=counter+1

    success = getFrame(sec)
