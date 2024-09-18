#dataset = 

# import cv2
# from PIL import Image
# from Color import get_limits
# import pandas as pd
# import numpy as np

# yellow = [0, 255, 255]
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# cap = cv2.VideoCapture(0)
# index = ["type", "hex", "R", "G", "B"]
# csv = pd.read_csv('data.csv', names=index, header=None)

# lowerLimit = np.array([0,48,80], dtype = np.uint8)
# upperLimit = np.array([20,255,255], dtype = np.uint8)

# def get_type(R, G, B):
#     minimum = 10000
#     for i in range(len(csv)):
#         d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
#         if d<=minimum:
#             minimum = d
#             tipe = csv.loc[i, "type"]
#     return tipe
   
# def draw_function(event, x, y, flags, param):
#         global b, g, r, x_pos, y_pos, clicked
#         x_pos = x
#         y_pos = y
#         b, g, r = frame[y,x]
#         b = int(b)
#         g = int(g)
#         r = int(r)

# while True:
#     ret, frame = cap.read()

#     height, width, ret = frame.shape

#     cx = int(width / 2)
#     cy = int(height / 2)

    # lowerLimit, upperLimit = get_limits(color = yellow)
    # hsvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsvimage, lowerLimit, upperLimit)

    # cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 3)
    # pixel_center = hsvimage[cy,cx]
    # b = int(pixel_center[0])
    # g = int(pixel_center[1])
    # r = int(pixel_center[2])
    # print(pixel_center)

    # text = get_type(r,g,b)

    # for i in range(len(csv)):
    #     minimum = 10000
    #     d = abs(r - csv.loc[i, "R"]) + abs(g - csv.loc[i, "G"]) + abs(b - csv.loc[i, "B"])
    #     if d<=minimum:
    #         minimum = d
    #         tipe = csv.loc[i, "type"]
            

    # text = type(tipe)
    # cv2.rectangle(frame, (20,20), (500,60), (232, 214,214), -1)
    # text = 'Your Personal Color is ' + get_type(r,g,b)
    # cv2.putText(frame, text, (50,50), 2, 0.8, (0,0,0), 2)
    # print(get_type(r,g,b))
    # print(b)
    # print(g)
    # print(r)
    
    # text = get_type(r, g, b)
    # print(text)

    # cv2.putText(hsvimage, text, (50,50), 2, 0.8, (255,255,255), 2, cv2.LINE_AA)
    # cv2.imshow("mask", mask)
    

    # -----------------------------------------------------------------------------------
    # cv2.rectangle(hsvimage, (20,20), (750, 60), (b,g,r), -1)

    # text = get_color(r,g,b)
    
    # cv2.putText(hsvimage, text, (50,50), 2, 0.8, (255,255,255), 2, cv2.LINE_AA)
    # cv2.putTeget_color(r,g,b))

    # if r + g + b >= 600:
    #     cv2.putText(hsvimage, text, (50,50), 2, 0.8, (0,0,0), 2, cv2.LINE_AA)

    # clicked = False
    # color = get_color(r,g,b)
    
    # -----------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------
    
    # mask = cv2.inRange(hsvimage, lowerLimit, upperLimit)
    # mask_ = Image.fromarray(mask)
    # bbox = mask_.getbbox()
    # print(bbox)
    # if bbox is not None:
    #     x1, y1, x2, y2 = bbox
    #     cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 5)
    # # -----------------------------------------------------------------------------------

    # cv2.putText(frame, colornum, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 5,(255,50,50), 4)
    # cv2.imshow('frame', frame)
    # key = cv2.waitKey(500)
    # if key == 32:
    #     cv2.waitKey()
    # if key == ord('q'):
    #     break

# cap.release()   
# cv2.destroyAllWindows()


import cv2
from PIL import Image
from Color import get_limits
import pandas as pd
import numpy as np

cap = cv2.VideoCapture(0)
index = ["type", "hex", "R", "G", "B"]
csv = pd.read_csv('data.csv', names=index, header=None)

def get_type(R, G, B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d<=minimum:
            minimum = d
            tipe = csv.loc[i, "type"]
    return tipe

while True:
    ret, frame = cap.read()

    height, width, ret = frame.shape

    cx = int(width / 2)
    cy = int(height / 2)

    hsvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 3)
    pixel_center = hsvimage[cy,cx]
    b = int(pixel_center[0])
    g = int(pixel_center[1])
    r = int(pixel_center[2])

    cv2.rectangle(frame, (20,20), (500,60), (232, 214,214), -1)
    text = 'Your Personal Color is ' + get_type(r,g,b)
    cv2.putText(frame, text, (50,50), 2, 0.8, (0,0,0), 2)
    print(get_type(r,g,b))
    print(b)
    print(g)
    print(r)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(500)
    if key == 32:
        cv2.waitKey()
    if key == ord('q'):
        break

cap.release()   
cv2.destroyAllWindows()
