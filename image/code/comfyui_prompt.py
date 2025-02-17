# Modify from ComfyUI example
# Reference: https://github.com/comfyanonymous/ComfyUI/blob/master/script_examples/websockets_api_example.py
import requests
#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint

import websocket  # Note: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse

from requests_toolbelt import MultipartEncoder

server_address = "127.0.0.1:8188"


def queue_prompt(prompt, client_id):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()


def get_image_data(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        image_data = {
            "content_type": response.info().get_content_type(),
            "data": response.read(),
        }
        return image_data


def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())


def get_images(ws, client_id, prompt):
    prompt_id = queue_prompt(prompt, client_id)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break  #Execution is done
        else:
            continue  #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images


def prompt_for_image_data(ws, client_id, prompt):
    """
    Execute prompt to get image data
    (only one image is returned)
    
    Return dictionary:
        data: binary image in byte
        content_type: string
    """
    prompt_id = queue_prompt(prompt, client_id)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break  #Execution is done
        else:
            continue  #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                for image in node_output['images']:
                    # only one image is returned, other images are ignored
                    image_data = get_image_data(image['filename'], image['subfolder'], image['type'])
                    return image_data

    return output_images


def upload_image_from_s3_url(s3_url, name, server_address, image_type="input", overwrite=False):
    response = requests.get(s3_url)
    response.raise_for_status()
    file_content = response.content

    multipart_data = MultipartEncoder(
        fields={
            'image': (name, file_content, 'image/png'),
            'type': image_type,
            'overwrite': str(overwrite).lower()
        }
    )

    data = multipart_data
    headers = {'Content-Type': multipart_data.content_type}
    request = urllib.request.Request("http://{}/upload/image".format(server_address), data=data, headers=headers)
    with urllib.request.urlopen(request) as response:
        return response.read()

prompt_text = """
{
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 8,
            "denoise": 1,
            "latent_image": [
                "5",
                0
            ],
            "model": [
                "4",
                0
            ],
            "negative": [
                "7",
                0
            ],
            "positive": [
                "6",
                0
            ],
            "sampler_name": "euler",
            "scheduler": "normal",
            "seed": 8566257,
            "steps": 20
        }
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.ckpt"
        }
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "batch_size": 1,
            "height": 512,
            "width": 512
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [
                "4",
                1
            ],
            "text": "masterpiece best quality girl"
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [
                "4",
                1
            ],
            "text": "bad hands"
        }
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": [
                "3",
                0
            ],
            "vae": [
                "4",
                2
            ]
        }
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": [
                "8",
                0
            ]
        }
    }
}
"""

if __name__ == "__main__":
    import random
    import base64

    client_id = str(uuid.uuid4())

    prompt = json.loads(prompt_text)
    # set the text prompt for our positive CLIPTextEncode
    prompt["6"]["inputs"]["text"] = "masterpiece best quality man"

    # set the seed for our KSampler node
    prompt["3"]["inputs"]["seed"] = random.randint(0, 1e10)

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    print("Prompt:")
    print(json.dumps(prompt, indent=2))
    print("\n\n")

    if prompt["input_image"]:
        upload_image_from_s3_url(prompt["input_image"], "input_image.png", server_address, "input", overwrite=True)
        # remove from prompt
        prompt.pop("input_image")

    images = get_images(ws, client_id, prompt)
    for node_id in images:
        for image_data in images[node_id]:
            print("Base64 Image:")
            print(base64.b64encode(image_data).decode("utf-8"))
            print("\n\n")

    # Commented out code to display the output images:

    # for node_id in images:
    #     for image_data in images[node_id]:
    #         from PIL import Image
    #         import io
    #         image = Image.open(io.BytesIO(image_data))
    #         image.show()
