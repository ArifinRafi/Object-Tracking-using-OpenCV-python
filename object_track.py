import cv2
cam = cv2.VideoCapture(0)
tracker = cv2.legacy_TrackerMOSSE.create()
ret, frame = cam.read()
bbox = cv2.selectROI('Tracking', frame, False)
tracker.init(frame, bbox)
def drawBox(frame, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x,y), ((x+w), (y+h)), (255, 0, 0), 3, 1)
    cv2.putText(frame, "Tracking", (75, 75), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2, 3)


while True:
    timer = cv2.getCPUTickCount()
    ret, frame = cam.read()
    ret, bbox = tracker.update(frame)

    if ret:
        drawBox(frame, bbox)
    else:
        cv2.putText(frame, "Lost", (75, 75), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2, 3)

    fps = cv2.getTickFrequency()/(cv2.getCPUTickCount()-timer)
    cv2.putText(frame, str(fps), (75,50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2, 3)
    cv2.imshow('Object Tracking', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

#Turning off the window
cam.release()
cv2.destroyAllWindows()
