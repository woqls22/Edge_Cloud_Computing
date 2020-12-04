 # -*- coding: euc-kr -*- 
import socket
import cv2
import numpy as np
from queue import Queue
from _thread import *
enclose_q = Queue()
recv_enclose_q = Queue()
def filter_img(img):
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
def bboxes(inp):
    img = inp
    start = time.time()
    curTime = time.time()
    img_final = inp
    img2gray = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY) #GRAY Image 8bit per pixel
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY) #threshold : distinguish background, object
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask) #bitwise
    ret, new_img = cv2.threshold(img_final, 180, 255, cv2.THRESH_BINARY)  # Nfor black text , cv.THRESH_BINARY_IV
    newimg = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY) #Gray Image converting
    _,contours, _ = cv2.findContours(newimg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # get contours
    #cv2.CHAIN_APPROX_NONE: All of contour point
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if h / w > 1.0 or w / h > 2.0:
            continue
        #if h>40 or w>70:
            #continue
        if y>150:
            continue
        cropped = img_final[y :y +  h , x : x + w]
        if(filter_img(cropped)):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img,"cropped", (x-50,y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
        else:
            continue
    return cropped
def send_threaded(Client_socket, addr, queue):
    print("Connected by : ", addr[0], " : ", addr[1])
    while True:
        try :
            data = Client_socket.recv(1024)
            if not data:
                print("Disconnected")
                break
            StringData = queue.get()
            Client_socket.send(str(len(StringData)).ljust(16).encode())
            Client_socket.send(StringData)
        except ConnectionResetError as e:
            print("Disconnected")
    Client_socket.close()
def webcam(queue):
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        if ret == False:
            continue
        frame = bboxes(frame)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        data = np.array(imgencode)
        stringData = data.tostring()
        queue.put(stringData)
        cv2.imshow('image', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
def recvall(sock, count):
    # byte string
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf
if __name__ == '__main__':
    SEND_HOST = '192.168.35.227' #CORE CLOUD
    SEND_PORT = 9999
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SEND_HOST, SEND_PORT))
    server_socket.listen()
    RECV_HOST = '192.168.35.87' #EDGE CLOUD
    RECV_PORT = 9998
    recv_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    recv_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    recv_server_socket.bind((RECV_HOST, RECV_PORT))
    recv_server_socket.listen()
    print('server start')
    start_new_thread(webcam, (enclose_q,))
    while True:
        print('wait')
        client_socket, addr = server_socket.accept()
        start_new_thread(send_threaded, (client_socket, addr, enclose_q,)) #preprocessed data recieve.
        conn,addr = recv_server_socket.accept() #waiting receiving...
        if(conn): 
            length = recvall(conn, 16)
            stringData = recvall(conn, int(length))
            data = np.fromstring(stringData, dtype = 'uint8') #receive driving info.
            if(conn): #loop ex
                break
        else:
            pass
    server_socket.close()
