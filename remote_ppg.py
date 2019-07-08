"""
    Remote PPG Algorithm and GUI. Measures a photoplethysmogram 
    in realtime from a webcame and allows direct comparison to 
    ppg obtained from "ground truth" finger pulse sensor.

    Written by Jimmy Newland as part of Rice University 
    research experience for teachers, summer 2019.

    http://jimmynewland.com
    newton@jayfox.net

"""
# OpenCV for camera access and frame reading
import cv2

# GUI and plotting tools
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from scipy import fftpack
import pyqtgraph.console

# Heartrate analysis package
import heartpy as hp
from heartpy.exceptions import BadSignalWarning

# Allows for filtering
from scipy import signal
from scipy.signal import butter, lfilter

# For file saving
import time
import array
from datetime import datetime
import csv
import io

# For pulse sensor
#import pulse_sensor as ps
import serial
from time import sleep

## Pulse Sensor Setup 
# http://www.jimmynewland.com/wp/about-jimmy/presentations/comparing-ppg-signals-open-vs-closed/
amped_comport = '/dev/cu.usbmodem1411301'
amped_baudrate = 115200
amped_serial_timeout = 1
now = datetime.now()

start_time = datetime.now()

ser = serial.Serial(amped_comport, amped_baudrate, \
    timeout=amped_serial_timeout)    # open serial port
## end PS Setup

## Qt GUI Setup
pg.mkQApp()
win = pg.GraphicsLayoutWidget()
camPen = pg.mkPen(width=10, color='y')
psPen = pg.mkPen(width=10,color='g')
win.setWindowTitle('Remote PPG')
# end Qt Setup

### OpenCV call to access webcam
cap = cv2.VideoCapture(0)

# Read from the webcam (OpenCV)
ret, frame = cap.read() 
h = frame.shape[0] # number of columns
w = frame.shape[1] # number of rows
aspect = h/w
## end OpenCV

#### GUI Setup
## Plot for image
imgPlot = win.addPlot(colspan=2)
imgPlot.getViewBox().setAspectLocked(True)
win.nextRow()

## Plot for camera intensity
camPlot = win.addPlot()
camBPMPlot = win.addPlot()
win.nextRow()

## Plot for pulse sensor intensity
psPlot = win.addPlot()
psBPMPlot = win.addPlot()

# ImageItem box for displaying image data
img = pg.ImageItem()
imgPlot.addItem(img)
imgPlot.getAxis('bottom').setStyle(showValues=False)
imgPlot.getAxis('left').setStyle(showValues=False)
imgPlot.getAxis('bottom').setPen(0,0,0)
imgPlot.getAxis('left').setPen(0,0,0)

win.show() # Display the window
#### end GUI Setup

# frequency sample in Hz
fs = 100

# Initalize
camData = np.random.normal(size=50)
camBPMData = np.zeros(50)

psData = np.random.normal(size=50)
psBPMData = np.zeros(50)

camPlot.getAxis('bottom').setStyle(showValues=False)
camPlot.getAxis('left').setStyle(showValues=False)

camBPMPlot.getAxis('bottom').setStyle(showValues=False)
camBPMPlot.setLabel('left','Cam BPM')

psBPMPlot.getAxis('bottom').setStyle(showValues=False)
psBPMPlot.setLabel('left','PS BPM')

# Used linspace instead of arange due to spacing errors
t = np.linspace(start=0,stop=5.0,num=50)
psTime = np.linspace(start=0,stop=5.0,num=50)

camCurve = camPlot.plot(t, camData, pen=camPen,name="Camera")
camPlot.setLabel('left','Cam Signal')

camBPMCurve = camBPMPlot.plot(t,camBPMData,pen=camPen,name="Cam BPM")

psCurve = psPlot.plot(t, psData, pen=psPen,name="Pulse Sensor")

psBPMCurve = psBPMPlot.plot(t, psBPMData, pen=psPen,name="PS BPM")

psPlot.getAxis('bottom').setStyle(showValues=False)
psPlot.getAxis('left').setStyle(showValues=False)
psPlot.setLabel('left','PS Signal')
ptr = 0

ret, frame = cap.read() # gets one frame from the webcam

# Convery image to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
numCols = gray.shape[0]
numRows = gray.shape[1]
        
middleRow = int(numRows/2)
middleCol = int(numCols/2)

boxH = int(numRows*0.15)
boxW = int(numCols*0.15)

box = pg.RectROI( (middleRow-boxH/2,middleCol-boxW/2), \
    (boxH,boxW), pen=9, sideScalers=True, centered=True)
imgPlot.addItem(box)

def get_data_ps():
    bpm = -1
    ibi = -1
    signal = -1
    serialRead = ser.readline()
    single_record = {}
    arduino_input = str(serialRead.strip())

    if arduino_input.count(",") == 2:
        bpm,ibi,signal = arduino_input.split(",")
        elapsed = (datetime.now() - start_time).total_seconds()

        single_record['pulseRate'] = bpm
        single_record['pulseWaveform'] = signal
        single_record['ibi'] = ibi
        single_record['time'] = elapsed
        return single_record
    else:
        return {"pulseRate":0,"pulseWaveform":0,"time":0,"ibi":0}

def setup_csv(csvStr=None):
    now = datetime.now()
    if csvStr is None:
        csvFileName = 'ppg_'+now.strftime("%Y-%m-%d_%I_%M_%S")
    else:
        csvFileName = csvStr+'_'+now.strftime("%Y-%m-%d_%I_%M_%S")
    headers = (u'ps_time'+','+u'ps_waveform'+','+u'ps_bpm'+','+u'cam_time'+','+u'cam_waveform'+','+u'cam_bpm')
    with io.open(csvFileName + '.csv', 'w', newline='') as f:
        f.write(headers)
        f.write(u'\n')
    return csvFileName

def save_to_csv(csvFileName, data):
    with io.open(csvFileName + ".csv", "a", newline="") as f:
        row = str(data['ps_time'])+","+str(data['ps_pulseWaveform'])+","+str(data['ps_pulseRate'])+"," \
             +str(data['cam_time'])+","+str(data['cam_pulseWaveform'])+","+str(data['cam_bpm'])
        f.write(row)
        f.write("\n") 

def update():
    global camData, camCurve, ptr, t, filename
    # Grab all the data from this frame
    image, signal = grabCam()
    data_ps = get_data_ps()

    ### get camera signal and subtract mean
    #sig = camData - np.mean(camData)

    ps_signal = data_ps['pulseWaveform']
    ps_bpm = data_ps['pulseRate']
    ps_time = data_ps['time']

    ### Python 2 vs Python 3 unicode encoding
    # Known issue
    if(isinstance(ps_signal,str)):
        ps_signal = int(ps_signal[:-1])
    else:
        ps_signal = int(ps_signal)

    if(isinstance(ps_bpm,str)):
        ps_bpm = int(ps_bpm[2:])
    else:
        ps_bpm = int(ps_bpm)
    ####
    
    ### heartpy
    # https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/quickstart.html#basic-example
    # https://www.researchgate.net/publication/328654252_Analysing_Noisy_Driver_Physiology_Real-Time_Using_Off-the-Shelf_Sensors_Heart_Rate_Analysis_Software_from_the_Taking_the_Fast_Lane_Project
    cam_bpm = camBPMData[-1]
    camSig = camData - np.mean(camData)
    try:
        working_data, measures = hp.process(camSig, 10.0)
    except BadSignalWarning:
        print("Bad signal")
    else:
        if(measures['bpm'] > 50 and measures['bpm'] < 120):
            cam_bpm = measures['bpm']
    ### end HeartPy

    # https://github.com/fxthomas/pg-examples/blob/master/linked_rois.py
    # PyQtGraph and OpenCV don't agree on whether pixels are row-major 
    # or vice versa.
    image = image.T[:, ::-1]
    img.setImage(image, autoLevels=True)

    camData[:-1] = camData[1:]  # shift data in the array one sample left
                            # (see also: np.roll)
    camData[-1] = signal

    camBPMData[:-1] = camBPMData[1:]
    camBPMData[:-1] = cam_bpm

    psData[:-1] = psData[1:]  # shift data in the array one sample left
                            # (see also: np.roll)
    psData[-1] = ps_signal

    psBPMData[:-1] = psBPMData[1:]
    psBPMData[-1] = ps_bpm

    t[:-1] = t[1:]
    t[-1] = (datetime.now() - start_time).total_seconds()

    psTime[:-1] = psTime[1:]
    psTime[-1] = ps_time
    
    # Package data to be saved to CSV.
    single_record = {}
    single_record['ps_time'] = ps_time
    single_record['ps_pulseRate'] = ps_bpm
    single_record['ps_pulseWaveform'] = ps_signal
    single_record['cam_pulseWaveform'] = camData[-1]
    single_record['cam_bpm'] =cam_bpm
    single_record['cam_time'] = t[-1]
    save_to_csv(filename, single_record)
    ## end CSV

    ptr += 1
    camCurve.setData(camData)
    camCurve.setPos(ptr, 0)

    camBPMCurve.setData(camBPMData)
    camBPMCurve.setPos(ptr, 0)

    psCurve.setData(psTime,psData)
    psCurve.setPos(ptr, 0)

    psBPMCurve.setData(psTime,psBPMData)
    psBPMCurve.setPos(ptr, 0)

    #print("cam:"+str(np.median(camBPMData))+", ps:"+str(psBPMData[-1]))
    print(psTime[-1],t[-1])


def grabCam():
    ret, frame = cap.read() # gets one frame from the webcam

    # Use OpenCV to convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create an resizeable ROI
    col,row = box.pos()
    row = int(row)
    col = int(col)
    
    x,y = box.size()
    x = int(x)
    y = int(y)

    roi = gray[row:row+y, col:col+x]
    ## end Roi

    # Find intensity (average or median or sum?)
    rowSum = np.sum(roi, axis=0)
    colSum = np.sum(rowSum, axis=0)
    allSum = rowSum + colSum

    intensity = np.median(np.median(allSum))
    
    return gray, intensity

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
tickTime = 1000/fs # how many milliseconds to wait.
timer.start(tickTime)

## Setup CSV File
filename=setup_csv()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()