# plaidml
# import plaidml.keras
# plaidml.keras.install_backend()

# packages
from keras.models import load_model
from keras.preprocessing import image

# import queue
import datetime
import numpy as np
from queue import Full, Empty
from multiprocessing import Process, Queue
import cv2

fname = 'croppedimg/{}.png'
save_imgs = False
HOST = '192.168.35.87'
PORT = 9999

class LabelingModule:
    def __init__(self):
        self.model1 = load_model('checker_model.h5')
        self.model2 = load_model('svhn_model.h5')
        self.image_queue = Queue(maxsize=3000)
        self.label_queue = Queue(maxsize=10)
        self.signal_queue = Queue()
        self.predict_process = Process(target=_predict, \
                                       args=(
                                       self.model1, self.model2, self.image_queue, self.label_queue, self.signal_queue))

    def run(self):
        self.predict_process.start()

    def close(self):
        self.signal_queue.put_nowait('stop')
        self.image_queue.close()
        self.label_queue.close()

    def new_tensor(self, tensor):
        try:
            self.image_queue.put(tensor)
        except Full:
            print('[LabelingModule] image_queue is full')

    def new_image(self, filename):
        tensor = self._img_to_tensor(filename)
        try:
            self.image_queue.put(tensor)
        except Full:
            print('[LabelingModule] image_queue is full')

    def _img_to_tensor(self, filename):
        img = image.load_img(filename, target_size=(48, 48))
        img_tensor = image.img_to_array(img)
        img_tensor = np.squeeze(img_tensor)
        img_tensor /= 255.
        img_tensor = img_tensor - img_tensor.mean()
        return img_tensor

def decode(output):
    if(output[0]==0):
        return 'Noise'
    else:
        if(output[1] == 3):
            return str(output[2])+str(output[3])+str(output[4])
        elif (output[1] == 4):
            return str(output[2]) + str(output[3]) + str(output[4])+'-'+ str(output[5])

def send_predict_result(HOST, PORT,message):
    # (address family) IPv4,  TCP
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # raspberry pi addr
    client_socket.connect((HOST, PORT))
    client_socket.sendall('Door Plate Detected : '+message.encode('utf-8'))

def _predict(model1, model2, input_queue, output_queue, signal_queue):
    print('predict process started.')

    index = 0
    while True:
        try:
            signal = signal_queue.get_nowait()
            if signal == 'stop':
                break
        except Empty:
            pass

        try:
            tensor = input_queue.get(timeout=-1)
        except Empty:
            continue
        tensor = np.array([tensor])
        has_number = model1.predict(tensor)[0]
        if int(has_number[0]) == 1:
            continue

        if save_imgs:
            img = cv2.cvtColor(tensor[0], cv2.COLOR_RGB2BGR)
            cv2.imwrite(fname.format(index), img)
            index += 1

        label_data = model2.predict(tensor)
        o1 = np.argmax(label_data[0])
        o2 = np.argmax(label_data[1])
        o3 = np.argmax(label_data[2])
        o4 = np.argmax(label_data[3])
        o5 = np.argmax(label_data[4])
        o6 = np.argmax(label_data[5])
        output = [o1, o2, o3, o4, o5, o6]
        print('[LabelingModule] predict result :', decode(output))
        send_predict_result(HOST,PORT)
        print(decode(output)+" : Sended To Edge")
