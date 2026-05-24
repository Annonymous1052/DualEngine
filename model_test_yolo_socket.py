from ultralytics import YOLO
from PIL import Image
import socket
import io
import base64
import numpy as np
import time
import sys
from threading import Thread, Lock
from queue import Queue
import torch
import argparse

parser = argparse.ArgumentParser(description='YOLO Socket Server for Image Classification')
parser.add_argument('--ip', type=str, default='192.168.0.2', help='Server IP address to bind to')
parser.add_argument('--port', type=int, default=5000, help='Server port number to bind to')
args = parser.parse_args()

print(f"Server configuration:")
print(f"IP Address: {args.ip}")
print(f"Port: {args.port}")
print('========================================')

def signal_handler(signal, frame):
    print("Ctrl+C")
    socket_image.close()  # close socket
    sys.exit(0)


def recvall(sock, count):
    buf = b''

    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def recv_img_from_client(sock_img, img_queue):
    latency = 0
    t = 0
    img_size = int(sock_img.recv(1024))
    sock_img.send(b'OK\n')
    print(img_size)
    while True:
        t += 1
        start = time.time()
        b64data = recvall(sock_img, img_size + 1)
        if b64data is None:
            print("break from recvall")
            img_queue.put(None)
            break

        end = time.time()
        img_queue.put(b64data[:-1])
        latency += end - start

    sock_img.close()

    print('========================================')
    print('receive time: ' + str(latency / t) + " s")
    print('========================================')


def inference_worker(sock_img, img_queue, model, send_lock):
    latency = 0
    encoding = 0
    t = 0

    while True:
        image_b64 = img_queue.get()
        if image_b64 is None:
            print("break from inference")
            break

        start = time.time()
        decoded_image = io.BytesIO(base64.b64decode(image_b64))
        opened_image = Image.open(decoded_image)
        encoding += np.round((time.time() - start), 2)

        start = time.time()
        model(opened_image, verbose=False)
        end = time.time()

        inference_time = np.round((end - start), 2)
        latency += inference_time
        t += 1

        with send_lock:
            sock_img.send(b'DONE\n')

    if t > 0:
        print('========================================')
        print('Number of received image: ' + str(t))
        print('Inference time: ' + str(latency / t) + " s")
        print('Encoding time: ' + str(encoding / t) + " s")
        print('========================================')


def get_top_5_classes(results, model):
    probabilities = results.probs.tolist()
    class_indices = range(len(probabilities))
    sorted_results = sorted(zip(class_indices, probabilities), key=lambda x: x[1], reverse=True)
    top_5_results = sorted_results[:5]
    top_5_with_names = [(model.names[int(cls)], conf) for cls, conf in top_5_results]
    return top_5_with_names


BUFFER_SIZE = 4096
HOST = args.ip
PORT = args.port

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
if device == 'cpu':
    exit()

print(f"Starting server on {HOST}:{PORT}")
print("waiting connection...")
socket_img = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_img.bind((HOST, PORT))
socket_img.listen(1)

try:
    socket_image, address = socket_img.accept()
    print(f"Client connected from {address}")

except Exception as e:
    print(e)
    exit()

finally:
    pass


def decode_base64_image(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    image_data = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


image_queue = Queue(maxsize=1000)
send_lock = Lock()
yolo_model = YOLO('yolov8x-cls.pt', verbose=False)
Thread(target=inference_worker, args=(socket_image, image_queue, yolo_model, send_lock)).start()
Thread(target=recv_img_from_client, args=(socket_image, image_queue)).start()
