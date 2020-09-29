import cv2
import numpy as np
import matplotlib as plt
import time
fname = "./croppedimg/"
index = 0
prevTime = 0
def filter_img(img):
    #이미지의 RGB값을 분석하여 찾는 실내 Tag가 맞는지 판별
    img = cv2.resize(img, (10,10))
    first = [0,0,0]
    for x_loc in range(0, 10):
        for y_loc in range(0, 10):
            bgr_value = img[x_loc,y_loc]
            first=first+bgr_value
    first[0] = first[0]/100
    first[1] = first[1]/100
    first[2] = first[2]/100
    blue = first[0]<200 and first[0]>120
    green = first[1]>120 and first[1]<210
    red = first[2]>130 and first[2]<230

    if(blue and green and red):
        return True
    else:
        return False
def bboxes(inp,prevTime):
    #Frame을 인자로 전달받음
    img = inp
    start = time.time()
    curTime = time.time()
    # img2gray = cv2.imread(fname,0)
    # img = cv2.namedWindow(img,cv2.WINDOW_NORMAL)
    # img = cv2.resizeWindow(img,600,600)
    img_final = inp
    # img_final = cv2.namedWindow(fname,cv2.WINDOW_NORMAL)
    # img_final = cv2.resizeWindow(fname,600,600)
    img2gray = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY) #GRAY Image 8bit per pixel
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY) #threshold : distinguish background, object
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask) #bitwise
    ret, new_img = cv2.threshold(img_final, 180, 255, cv2.THRESH_BINARY)  # Nfor black text , cv.THRESH_BINARY_IV
    newimg = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY) #Gray Image converting
    #newimg = cv2.GaussianBlur(newimg, (3,3),0)

    # remove noise from image
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,1))
    # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    #dilated = cv2.dilate(newimg, kernel, iterations=1)  # dilate
   # erode = cv2.erode(newimg, kernel)
    _,contours, _ = cv2.findContours(newimg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # get contours
    #cv2.CHAIN_APPROX_NONE: 모든 컨투어 포인트를 반환
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)
        # remove small false positives that aren't textq
        # text인식하기. width, height
        #if w > 50 or h > 35 or w<25:
            #continue
        if h / w > 1.0 or w / h > 2.0:
            continue
        #if h>40 or w>70:
            #continue
        if y>150:
            continue
        cropped = img_final[y :y +  h , x : x + w]
        # draw rectangle around contour on original image
        if(filter_img(cropped)):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img,"cropped", (x-50,y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
            #cropped = img_final[y :y +  h , x : x + w]
        else:
            continue
    img = cv2.resize(img, (720, 380))
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)
    #print ("Time {0} ".format(sec))
    #print ("Estimated fps {0} ".format(fps))
    str1 = ("FPS : {0}".format(int(fps)))
    cv2.putText(img, str1, (0, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0),1)
    cv2.imshow('captcha_result', img)
    return prevTime

cap = cv2.VideoCapture(0) #동영상 파일 읽어옴
while (cap.isOpened()):
    ret, inp = cap.read() #프레임을 읽어옴, 읽어온 프레임을 인자로 bboxes 전달
    if(ret): #success boolean
        prevTime = bboxes(inp, prevTime)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Terminate Process..")
        break
cap.release() #파일 닫아줌