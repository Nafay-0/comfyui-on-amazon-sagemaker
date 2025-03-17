import base64
import json
import logging
import io
import os
import requests
import flask
import websocket  # Note: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
from comfyui_prompt import prompt_for_image_data, upload_image_from
from PIL import Image

# Define Logger
logger = logging.getLogger()
logging.basicConfig()
logger.setLevel(logging.INFO)
app = flask.Flask(__name__)
ws = None
client_id = None

# environment variable to set jpeg quality
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", 90))

# environment variable to print HTTP header of requests
DEBUG_HEADER = os.getenv("DEBUG_HEADER", "False").lower() in ("true", "1", "t")

# contants for comfyui server
SERVER_ADDRESS = "127.0.0.1:8188"
URL_PING = f"http://{SERVER_ADDRESS}"


@app.route("/ping", methods=["GET"])
def ping():
    """
    Check the health of the ComfyUI local server is responding

    Returns a 200 status code if success, or a 500 status code if there is an error.

    Returns:
        flask.Response: A response object containing the status code and mimetype.
    """
    # Check if the local server is responding, set the status accordingly
    r = requests.head(URL_PING, timeout=5)
    status = 200 if r.ok else 500

    # Return the response with the determined status code
    return flask.Response(response="\n", status=status, mimetype="application/json")


def get_image_name(prompt_dict):
    for i in prompt_dict:
        if isinstance(prompt_dict[i], str):
            continue
        if "inputs" in prompt_dict[i]:
            if (
                    prompt_dict[i]["class_type"] == "LoadImage"
                    and "image" in prompt_dict[i]["inputs"]
            ):
                return prompt_dict[i]["inputs"]["image"]
    return None


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Handle prediction requests and return all generated images.
    Returns a JSON array containing image data and content types for all generated images.
    """
    global ws, client_id
    if ws is None or client_id is None:
        client_id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(SERVER_ADDRESS, client_id))

    if DEBUG_HEADER:
        print(flask.request.headers)

    # get prompt from request body regardless of content type
    prompt = flask.request.get_json(silent=True, force=True)

    logger.info("Prompt received in the request")
    prompt_str = json.dumps(prompt, indent=2)
    logger.info(prompt_str)

    # if image input is provided, upload it to comfyui server
    if prompt.get("input_image"):
        image_data = prompt["input_image"]
        image_data = base64.b64decode(image_data)
        filename = "input1.png"
        res = upload_image_from(image_data, filename, SERVER_ADDRESS)
        prompt.pop("input_image")
    else:
        logger.info("No image received in the request")

    # Get all generated images
    image_data_arr = prompt_for_image_data(ws, client_id, prompt)
    logger.info(f"Number of images generated: {len(image_data_arr)}")

    # Process each image according to accept headers
    processed_images = []
    accept_jpeg = "image/jpeg" in flask.request.accept_mimetypes

    for image_data in image_data_arr:
        current_image = {}

        # Convert PNG to JPEG if requested and possible
        if accept_jpeg and image_data.get("content_type") == "image/png":
            png_image = Image.open(io.BytesIO(image_data.get("data")))
            rgb_image = png_image.convert("RGB")
            jpeg_bytes = io.BytesIO()
            rgb_image.save(jpeg_bytes, format="jpeg", optimize=True, quality=JPEG_QUALITY)
            current_image = {
                "data": base64.b64encode(jpeg_bytes.getvalue()).decode('utf-8'),
                "content_type": "image/jpeg"
            }
        else:
            current_image = {
                "data": base64.b64encode(image_data.get("data")).decode('utf-8'),
                "content_type": image_data.get("content_type")
            }
        processed_images.append(current_image)

    # Return array of all processed images
    return flask.Response(
        response=json.dumps({
            "images": processed_images,
            "total_images": len(processed_images)
        }),
        status=200,
        mimetype="application/json"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
