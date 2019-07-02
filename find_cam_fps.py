import cv2
import time
 
video = cv2.VideoCapture(0)

num_frames = 120

#print(num_frames)
 
start = time.time()
     
for i in range(0, num_frames) :
    ret, frame = video.read()
 
     
end = time.time()
 
seconds = end - start
#print(seconds)
 
fps  = num_frames / seconds

print(fps)
 
video.release()