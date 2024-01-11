#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint
import os
import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from flask import Flask, request, jsonify, render_template, session, abort
from flask_socketio import SocketIO, join_room, leave_room,send, emit
import secrets

from PIL import Image
import io
import base64

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileCreatedHandler(FileSystemEventHandler):
    def on_created(self, event):
        # 处理文件创建完成的逻辑
        file_path = event.src_path
        file_name = os.path.basename(file_path)
        print(f"New file created: {file_name}")
        
        time.sleep(.1)
        with open(file_path, 'rb') as fr:
            image_data=fr.read()
            b64img=base64.b64encode(image_data).decode('utf-8')
            socketio.emit('server_response',{'b64img':b64img})

folder_to_watch = "/home/admin/ComfyUI/output/dragnuwa"  # 要监控的文件夹路径

event_handler = FileCreatedHandler()  # 创建我们刚才定义的自定义处理类的实例
observer = Observer()
observer.schedule(event_handler, folder_to_watch, recursive=False)
observer.start()

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}

prompt={}
with open('./workflow_api.json') as fr:
    prompt = json.load(fr)

ws = websocket.WebSocket()
ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

app = Flask(__name__, template_folder=os.path.abspath('.'), static_folder='assets')
app.secret_key = secrets.token_hex(16)

socketio = SocketIO(app, cors_allowed_origins='*')
connected_sids = set()  # 存放已连接的客户端

#后端程序
lockroom='None'
@socketio.on('connect')
def on_connect():
    connected_sids.add(request.sid)
    print(f'{request.sid} 已连接')
    socketio.start_background_task(background_thread_heartbeat)

@socketio.on('disconnect')
def on_disconnect():
    connected_sids.remove(request.sid)
    print(f'{request.sid} 已断开')

@socketio.on('message')
def handle_message(message):
    """收消息"""
    print(f'message:{request.sid} {message}')
    json.loads(message)

@socketio.on('camera_poses')
def handle_message(camera_poses):
    print(f'camera_poses:{request.sid} {camera_poses}')
    img=camera_poses["img"]
    trajs=json.loads(camera_poses["trajs"])
    prompt["7"]["inputs"]["image_data"] = img

    prompt["6"]["inputs"]["tracking_points"] = camera_poses["trajs"]
    images = get_images(ws, prompt)

@socketio.on('server_reconnect')
def server_reconnect(message):
    print(f'server_reconnect:{request.sid} {message}')
    join_room(message['roomid'])

def background_thread_heartbeat():
    global lockroom
    while True:
        socketio.emit('server_response',{'lockroom':lockroom})
        socketio.sleep(5)


@app.route('/')
def index():
    session['user'] = None
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5017, debug=True, allow_unsafe_werkzeug=True)
    #app.run(host='0.0.0.0', port=5017)