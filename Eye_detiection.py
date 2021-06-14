import cv2
import dlib
import numpy as np


#center_check = 0
#center = np.array([])
#avg=0

patterns = ['left','right']
pattern_length=0
p='straight'
def shape_to_np(shape, dtype="int"):
   # initialize the list of (x, y)-coordinates
   coords = np.zeros((68, 2), dtype=dtype)
   # loop over the 68 facial landmarks and convert them
   # to a 2-tuple of (x, y)-coordinates
   for i in range(0, 68):
      coords[i] = (shape.part(i).x, shape.part(i).y)
   # return the list of (x, y)-coordinates
   return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


def center_checking(thresh,mid,img,shapeo):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts,ket=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        #cy = int(M['m01']/M['m00'])
        cx += mid
        if(cx>0):
            left = abs(shape[42][0]-cx)
            center=np.append(center,left)
            return 1
        return 0
            
        #cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass
    
    
def contouring(thresh, mid, img,shape, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
            if(cx>0 and cy>0):
                left = abs(shape[42][0]-cx)
                right = abs(shape[45][0]-cx)

        else:
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
            if(cx>=0 and abs(shape[36][0]-cx)<=15):
                print("right")
                return 'right'
            

            elif(cx>=0 and abs(shape[39][0]-cx)<=14):
                print("left")
                return 'left'
            
            else:
                
                return 'straight'
                
    except:
        pass

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
temp_pattern=patterns.copy()
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:

        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        #print(shape[42][0])
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3
        thresh = cv2.bitwise_not(thresh)
        temp_p=p
        p=contouring(thresh[:, 0:mid], mid, img,shape)
        contouring(thresh[:, mid:], mid, img,shape, True)
        if(temp_p=='straight' and p!='straight' and len(temp_pattern)!=0):
            if(p==temp_pattern.pop()):
                pattern_length+=1
            else:
                print("WRONG")
                pattern_length=0
                temp_pattern=patterns.copy()
        
    if(pattern_length==2):
        cv2.putText(img, "unLocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(img, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('image', img)
    #cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()