import cv2
from xtra_model import xtra_model
import math
import logging

logger = logging.getLogger('run')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


cap=cv2.VideoCapture('sample.mp4')

_, image = cap.read()
pose = xtra_model()


def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


while True:
    _, frame = cap.read()
    _, frame = cap.read()
    _, frame = cap.read()
    _, frame = cap.read()
    label_keypoint = pose.get_feedback(frame)
    
    neck = label_keypoint.get('Neck')    
    rwri = label_keypoint.get('Rwri')
    lwri = label_keypoint.get('Lwri')
    
        
    radius = 2
    color = (255, 0, 0)
    thickness = 2
    org = (200, 50)
    
    if(rwri and lwri):
        image = cv2.circle(frame, neck, radius, color, thickness)
        image = cv2.circle(frame, rwri, radius, color, thickness)
        image = cv2.circle(frame, lwri, radius, color, thickness)

        image = cv2.line(image, neck, rwri, color, thickness) 
        image = cv2.line(image, neck, lwri, color, thickness) 
        
        angle = getAngle(rwri, neck, lwri)
        ang = round(angle, 2)
        message = "Angle : "+str(ang)+" degree"
        logger.info(message)
        image = cv2.putText(image, message, org, cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), thickness, cv2.LINE_AA)
    else:

        message = "Hands not not visible. Make sure both the wrist are visible in the frame."
        logger.error(message)

    cv2.imshow("capture",frame) 
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

    _, frame = cap.read()
    _, frame = cap.read()

    