# plaidml
# import plaidml.keras
# plaidml.keras.install_backend()

# packages
from keras.models import load_model
from keras.preprocessing import image

# import queue
import numpy as np
from queue import Full, Empty
from multiprocessing import Process, Queue

class LabelingModule:
    def __init__(self):
        # self.model1 = load_model('svhn_model.h5')
        self.model2 = load_model('svhn_model.h5')
        self.image_queue = Queue(maxsize=3000)
        self.label_queue = Queue(maxsize=10)
        self.signal_queue = Queue()
        self.predict_process = Process(target=_predict, \
            args=(self.model2, self.image_queue, self.label_queue, self.signal_queue))

    def run(self):
        self.predict_process.start()

    def close(self):
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

def _predict(model, input_queue, output_queue, signal_queue):
    print('predict process started.')
    while True:
        try:
            signal = signal_queue.get_nowait()
            if signal == 'stop':
                break
        except Empty:
            pass
        
        tensor = input_queue.get(timeout=-1)
        dat = model.predict(np.array([tensor]))
        o1 = np.argmax(dat[0])
        o2 = np.argmax(dat[1])
        o3 = np.argmax(dat[2])
        o4 = np.argmax(dat[3])
        o5 = np.argmax(dat[4])
        o6 = np.argmax(dat[5])
        output = [o1, o2, o3, o4, o5, o6]
        print('[LabelingModule] predict result :', output)
