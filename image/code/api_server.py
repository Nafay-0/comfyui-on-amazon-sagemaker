import base64
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
    Handle prediction requests by transforming the input data and returning the
    transformed data as a CSV string.

    This function checks if the request content type is supported (text/csv),
    and if so, decodes the input data, transforms it using the transform_fn
    function, and returns the transformed data as a CSV string. If the content
    type is not supported, a 415 status code is returned.

    Returns:
        flask.Response: A response object containing the transformed data,
                        status code, and mimetype.
    """
    global ws, client_id
    if ws is None or client_id is None:
        client_id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(SERVER_ADDRESS, client_id))  # nosemgrep: detect-insecure-websocket

    if DEBUG_HEADER:
        print(flask.request.headers)

    # get prompt from request body regardless of content type
    prompt = flask.request.get_json(silent=True, force=True)

    # if image input is provided, upload it to comfyui server
    if prompt.get("input_image"):
        image_data = prompt["input_image"]
        image_data = base64.b64decode(image_data)
        filename = "input1.png"
        res = upload_image_from(image_data, filename, SERVER_ADDRESS)
        # remove the fields input_image  from prompt
        prompt.pop("input_image")
    else:
        logger.info("No image recieved in the request")
    image_data = prompt_for_image_data(ws, client_id, prompt)

    # convert png to jpeg if it is allowed from accept header
    accept_jpeg = "image/jpeg" in flask.request.accept_mimetypes
    if accept_jpeg and image_data.get("content_type") == "image/png":
        png_image = Image.open(io.BytesIO(image_data.get("data")))
        rgb_image = png_image.convert("RGB")
        jpeg_bytes = io.BytesIO()
        rgb_image.save(jpeg_bytes, format="jpeg", optimize=True, quality=JPEG_QUALITY)
        image_data["data"] = jpeg_bytes.getvalue()
        image_data["content_type"] = "image/jpeg"

    return flask.Response(
        response=image_data.get("data", ""),
        status=200,
        mimetype=image_data.get("content_type", "text/plain"),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)