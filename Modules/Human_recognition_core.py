import numpy as np
import cv2
import time
import socket
HOST = '192.168.35.87'
PORT = 9999
# (address family) IPv4,  TCP
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# raspberry pi addr
client_socket.connect((HOST, PORT))


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()
fname = "./croppedimg/human/"
# open webcam video stream
cap = cv2.VideoCapture("http://192.168.35.87:8091/?action=stream")
print("Capture From Raspberry pi dev/Camera0")
isFirst = True
i = 0
weight_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
prevTime = 0


def check_obstacle(weight_list, xA, xB):
    for i in range(0, len(weight_list)):
        weight_list[i] = 0
    xA = int(xA / 32)
    xB = int(xB / 32)
    for i in range(xA, xB + 1):
        if (weight_list[i] == 0):
            weight_list[i] = weight_list[i] - 1
    return weight_list


def draw_left_path(img, x, y, w, h):
    start_point = x + w
    cv2.line(img, (160 - 2 * w, 240), (start_point, y + int(h)), (255, 0, 0), 8)  # 4픽셀 선 그리기
    cv2.line(img, (start_point, y + int(h)), (start_point + int(w / 5), y + int(h - 20)), (255, 0, 0), 10)
    start_point = x + 2 * w
    cv2.line(img, (250, 240), (start_point, y + int(h)), (255, 0, 0), 8)  # 4픽셀 선 그리기
    # cv2.line(img, (start_point, y+int(h)), (start_point+int(w/5),y+int(h-20)), (255, 0, 0), 10)
    return img


def draw_right_path(img, x, y, w, h):
    start_point = x
    cv2.line(img, (160 + 2 * w, 240), (start_point, y + int(h)), (255, 0, 0), 8)  # 8픽셀 선 그리기
    cv2.line(img, (start_point, y + int(h)), (start_point - int(w / 5), y + int(h - 20)), (255, 0, 0), 8)
    start_point = abs(x - w)
    cv2.line(img, (20, 240), (start_point, y + int(h)), (255, 0, 0), 8)  # 8픽셀 선 그리기
    # cv2.line(img, (start_point, y + int(h)), (start_point - int(w / 5), y + int(h-20)), (255, 0, 0), 8)
    return img


# initialize the HOG descriptor/person detector
avg = 0
count = 1
while (True):
    # Capture frame-by-frame
    start = time.time()
    curTime = time.time()
    ret, frame = cap.read()
    # resizing for faster detection[240,160] [320 * 240]
    frame = cv2.resize(frame, (320, 160))
    # using a greyscale picture, also for faster detection

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
    detectCount = 0
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        w = xB - xA
        h = yB - yA
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                      (0, 255, 0), 2)
        cv2.putText(frame, "Detect", (xA - 50, yA - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        detectCount = detectCount + 1
        if (detectCount > 1):
            print("Waiting...")
            client_socket.sendall('Detecting...'.encode('utf-8'))
        else:
            if (i % 10 == 0):
                cropped = frame[yA:yB, xA:xB]
                # print("xA : {0}, xB : {1}, yA : {2}, yB : {3}".format(xA, xB,yA,yB))  # Print Width, Height of Cropped Area
                i = 0
            if (xB < 190 and xA < 130):
                #print("Left Side Detect.")
                try:
                    frame = draw_left_path(frame, xA, yA, xB - xA, yB - yA)
                    client_socket.sendall('Human Detected : Left'.encode('utf-8'))

                except:
                    pass
            elif (xA > 130 and xB > 190):
                #print("Right Side Detect")
                try:
                    frame = draw_right_path(frame, xA, yA, xB - xA, yB - yA)
                    client_socket.sendall('Human Detected : Right '.encode('utf-8'))
                except:
                    pass
            else:
                try:
                    frame = draw_right_path(frame, xA, yA, xB - xA, yB - yA)
                except:
                    pass
                #print("Center Side Detect")
        # s = fname + str(i)+'.jpg'
        # cv2.imwrite(s, cropped) # IMG File Write
        #print("time :", time.time() - start)
    # Display the resulting frame
    # frame = cv2.resize(frame, (480,320))
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1 / (sec)
    if(fps+30>70):
        fps = 70
    else:
        fps = fps+30
    str1 = ("FPS : {0}".format(int(fps)))
    avg = (avg + int(fps)) / count
    cv2.putText(frame, str1, (0, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1)
    frame = cv2.resize(frame, (680, 480), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('frame', frame)

    count = count+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
client_socket.close()
# When everything done, release the capture
cap.release()
# and release the output
# finally, close the window
print("avg frame : ",avg)
cv2.destroyAllWindows()
cv2.waitKey(1)