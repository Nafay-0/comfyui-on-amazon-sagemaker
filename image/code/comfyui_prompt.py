import websocket  # Note: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from requests_toolbelt import MultipartEncoder
import urllib.request
import requests
import io

server_address = "127.0.0.1:8188"


def convert_prompt_format(prompt):
    # check if prompt is a string
    if isinstance(prompt, str):
        prompt = json.loads(prompt)

    # Create an empty dictionary to hold the converted format
    converted_prompt = {}

    # Iterate over each item in the original prompt
    for key, value in prompt.items():
        if isinstance(value, dict) and "class_type" in value:
            # If the item is a dictionary and contains class_type, it's part of the expected structure
            # Copy the dictionary with its class_type and inputs
            converted_prompt[key] = {
                "class_type": value["class_type"],
                "inputs": value["inputs"]
            }
        elif isinstance(value, dict):
            # If it's not the expected structure, we assume it's a general key-value pair (like seed, positive_prompt, negative_prompt)
            converted_prompt[key] = value

    # Return the converted dictionary
    return converted_prompt


def queue_prompt(prompt, client_id):
    prompt = convert_prompt_format(prompt)
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
                    break  # Execution is done
        else:
            continue  # previews are binary data

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
    Execute prompt to get image data for all generated images.

    Returns:
        list: List of dictionaries containing image data and content type
    """
    prompt_id = queue_prompt(prompt, client_id)['prompt_id']
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break  # Execution is done
        else:
            continue  # previews are binary data

    history = get_history(prompt_id)[prompt_id]
    image_data_arr = []

    # Collect all images from all output nodes
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image_data(image['filename'], image['subfolder'], image['type'])
                image_data_arr.append(image_data)

    return image_data_arr


def upload_image_from(image_data, name, server_address, image_type="input", overwrite=True):
    """
    Args:
        image_data (bytes): The image data to upload.
        name (str): The name to assign to the uploaded image.
        server_address (str): The server endpoint for uploading images.
        image_type (str, optional): The type of image. Defaults to "input".
        overwrite (bool, optional): Whether to overwrite the image if it exists. Defaults to False.

    Returns:
        str: The response from the server.
    """

    # Step 2: Prepare multipart form data
    multipart_data = MultipartEncoder(
        fields={
            'image': (name, image_data, 'image/png'),  # Change MIME type if needed
            'type': image_type,
            'overwrite': str(overwrite).lower()
        }
    )

    # Step 3: Send POST request
    upload_url = f"http://{server_address}/upload/image"  # Ensure this endpoint is correct
    headers = {'Content-Type': multipart_data.content_type}

    req = urllib.request.Request(upload_url, data=multipart_data, headers=headers)
    with urllib.request.urlopen(req) as response:
        return response.read().decode('utf-8')  # Decode response if it's in bytes


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
#
# if __name__ == "__main__":
#     import random
#     import base64
#
#     client_id = str(uuid.uuid4())
#
#     prompt = json.loads(prompt_text)
#     # set the text prompt for our positive CLIPTextEncode
#     prompt["6"]["inputs"]["text"] = "masterpiece best quality man"
#
#     # set the seed for our KSampler node
#     prompt["3"]["inputs"]["seed"] = random.randint(0, 1e10)
#
#     ws = websocket.WebSocket()
#     ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
#     print("Prompt:")
#     print(json.dumps(prompt, indent=2))
#     print("\n\n")
#
#     if prompt["input_image"]:
#         upload_image_from(prompt["input_image"], "input_image.png", server_address, "input", overwrite=True)
#         # remove from prompt
#         prompt.pop("input_image")
#
#     images = get_images(ws, client_id, prompt)
#     for node_id in images:
#         for image_data in images[node_id]:
#             print("Base64 Image:")
#             print(base64.b64encode(image_data).decode("utf-8"))
#             print("\n\n")
#
#     # Commented out code to display the output images:
#
#     # for node_id in images:
#     #     for image_data in images[node_id]:
#     #         from PIL import Image
#     #         import io
#     #         image = Image.open(io.BytesIO(image_data))
#     #         image.show()
