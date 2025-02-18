import json
import boto3
import logging
import random
import base64
import io
import os
# Define Logger
logger = logging.getLogger()
logging.basicConfig()
logger.setLevel(logging.INFO)

sagemaker_client = boto3.client("sagemaker-runtime")


def update_seed(prompt_dict, seed=None):
    """
    Update the seed value for the KSampler node in the prompt dictionary.

    Args:
        prompt_dict (dict): The prompt dictionary containing the node information.
        seed (int, optional): The seed value to set for the KSampler node. If not provided, a random seed will be generated.

    Returns:
        dict: The updated prompt dictionary with the seed value set for the KSampler node.
    """
    # set seed for KSampler node
    for i in prompt_dict:
        # if node is str skip
        if isinstance(prompt_dict[i], str):
            continue
        if "inputs" in prompt_dict[i]:
            if (
                    prompt_dict[i]["class_type"] == "KSampler"
                    and "seed" in prompt_dict[i]["inputs"]
            ):
                if seed is None:
                    prompt_dict[i]["inputs"]["seed"] = random.randint(0, int(1e10))
                else:
                    prompt_dict[i]["inputs"]["seed"] = int(seed)
    return prompt_dict


def update_image_dimensions(prompt_dict, width, height):
    """
    Update the image dimensions in the prompt dictionary for the latent image node.

    Args:
        prompt_dict (dict): The prompt dictionary containing the node information.
        width (int): The new width value.
        height (int): The new height value.

    Returns:
        dict: The updated prompt dictionary with the new image dimensions.
    """
    for node_id in prompt_dict:
        node = prompt_dict[node_id]
        # if node is str skip
        if isinstance(node, str):
            continue
        if node.get("class_type") == "EmptySD3LatentImage" and "inputs" in node:
            node["inputs"]["width"] = int(width)
            node["inputs"]["height"] = int(height)

        if node.get("class_type") == "EmptyLatentImage" and "inputs" in node:
            node["inputs"]["width"] = int(width)
            node["inputs"]["height"] = int(height)
    return prompt_dict


def update_Sampler_details(prompt_dict, steps=20, denoise=1, cfg=8, sampler_name="euler"):
    for i in prompt_dict:
        if isinstance(prompt_dict[i], str):
            continue
        if "inputs" in prompt_dict[i]:
            if (
                    prompt_dict[i]["class_type"] == "KSampler"
                    and "steps" in prompt_dict[i]["inputs"]
            ):
                prompt_dict[i]["inputs"]["steps"] = steps
                prompt_dict[i]["inputs"]["denoise"] = denoise
                prompt_dict[i]["inputs"]["cfg"] = cfg
                prompt_dict[i]["inputs"]["sampler_name"] = sampler_name

    return prompt_dict


def update_prompt_text(prompt_dict, positive_prompt, negative_prompt):
    """
    Update the prompt text in the given prompt dictionary.

    Args:
        prompt_dict (dict): The dictionary containing the prompt information.
        positive_prompt (str): The new text to replace the positive prompt placeholder.
        negative_prompt (str): The new text to replace the negative prompt placeholder.

    Returns:
        dict: The updated prompt dictionary.
    """
    # replace prompt text for CLIPTextEncode node
    for i in prompt_dict:
        if isinstance(prompt_dict[i], str):
            continue
        if "inputs" in prompt_dict[i]:
            if (
                    prompt_dict[i]["class_type"] == "CLIPTextEncode"
                    and "text" in prompt_dict[i]["inputs"]
            ):
                if prompt_dict[i]["inputs"]["text"] == "POSITIVE_PROMT_PLACEHOLDER":
                    prompt_dict[i]["inputs"]["text"] = positive_prompt
                elif prompt_dict[i]["inputs"]["text"] == "NEGATIVE_PROMPT_PLACEHOLDER":
                    prompt_dict[i]["inputs"]["text"] = negative_prompt
    return prompt_dict


def update_tensors_file_name(prompt_dict, tensors_file_name):
    # node name CheckpointLoaderSimple
    if tensors_file_name is None:
        return prompt_dict

    for i in prompt_dict:
        if isinstance(prompt_dict[i], str):
            continue
        if "inputs" in prompt_dict[i]:
            if (
                    prompt_dict[i]["class_type"] == "CheckpointLoaderSimple"
                    and "ckpt_name" in prompt_dict[i]["inputs"]
            ):
                prompt_dict[i]["inputs"]["ckpt_name"] = tensors_file_name
    return prompt_dict


def update_input_image_name(prompt_dict, input_image_name):
    # node name CheckpointLoaderSimple
    if input_image_name is None:
        return prompt_dict

    for i in prompt_dict:
        if isinstance(prompt_dict[i], str):
            continue
        if "inputs" in prompt_dict[i]:
            if (
                    prompt_dict[i]["class_type"] == "LoadImage"
                    and "image" in prompt_dict[i]["inputs"]
            ):
                prompt_dict[i]["inputs"]["image"] = input_image_name
    return prompt_dict


def get_image_from_url(url):
    """
    Get the image data from the provided URL.

    Args:
        s3 url (str): The URL to fetch the image data from.

    Returns:
        bytes: The image data in bytes.
    """
    # fetch image from boto client s3 url
    boto3_client = boto3.client("s3")
    bucket_name = url.split("/")[2]
    key = "/".join(url.split("/")[3:])
    response = boto3_client.get_object(Bucket=bucket_name, Key=key)
    file_content = io.BytesIO(response["Body"].read())
    file_name = key.split("/")[-1]
    return file_content, file_name


def invoke_from_prompt(prompt_file, positive_prompt, negative_prompt, seed=None, width=1024, height=1024,
                       steps=20, denoise=1, cfg=8, sampler_name="euler", tensors_file_name=None, image_input=None):
    """
    Invokes the SageMaker endpoint with the provided prompt data.

    Args:
        image_input:  The image input to be used in the prompt data.
        tensors_file_name:  The tensors file name to be used in the prompt data.
        sampler_name:  The sampler name to be used in the prompt data.
        cfg:  The cfg value to be used in the prompt data.
        denoise:  The denoise value to be used in the prompt data.
        steps:  The steps value to be used in the prompt data.
        prompt_file (str): The path to the JSON file in ./workflow/ containing the prompt data.
        positive_prompt (str): The positive prompt to be used in the prompt data.
        negative_prompt (str): The negative prompt to be used in the prompt data.
        seed (int, optional): The seed value for randomization. Defaults to None.
        width (int, optional): The width of the output image. Defaults to 1024.
        height (int, optional): The height of the output image. Defaults to 1024.

    Returns:
        dict: The response from the SageMaker endpoint.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    logger.info("prompt: %s", prompt_file)

    # read the prompt data from json file
    with open("./workflow/" + prompt_file) as prompt_file:
        prompt_text = prompt_file.read()
    prompt_dict = json.loads(prompt_text)
    prompt_dict = update_seed(prompt_dict, seed)
    prompt_dict = update_prompt_text(prompt_dict, positive_prompt, negative_prompt)
    prompt_dict = update_image_dimensions(prompt_dict, width, height)
    prompt_dict = update_Sampler_details(prompt_dict, steps, denoise, cfg, sampler_name)
    prompt_dict = update_tensors_file_name(prompt_dict, tensors_file_name)
    if image_input:
        url = image_input
        image_data, file_name = get_image_from_url(url)
        # add a new field to the prompt_dict
        prompt_dict["input_image"] = base64.b64encode(image_data.getvalue()).decode("utf-8")
        prompt_dict["input_image_name"] = file_name
        prompt_dict = update_input_image_name(prompt_dict, file_name)

    prompt_text = json.dumps(prompt_dict)

    endpoint_name = os.environ["ENDPOINT_NAME"]
    content_type = "application/json"
    accept = "*/*"
    payload = prompt_text
    payload_str = json.dumps(payload, indent=4)
    logger.info(f"Final payload to invoke sage maker endpoint: {payload_str}")
    logger.info(json.dumps(payload, indent=4))
    response = sagemaker_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Accept=accept,
        Body=payload,
    )
    return response


def lambda_handler(event: dict, context: dict):
    """
    Lambda function handler for processing events.

    Args:
        event (dict): The event from lambda function URL.
        context (dict): The runtime information of the Lambda function.

    Returns:
        dict: The response data for lambda function URL.
    """
    logger.info("Event:")
    logger.info(json.dumps(event, indent=2))
    request = json.loads(event["body"])

    try:
        prompt_file = request.get("prompt_file", "SDXL.json")
        positive_prompt = request["positive_prompt"]
        negative_prompt = request.get("negative_prompt", "")
        image_input = request.get("image_input", None)
        width = request.get("width", 1024)
        height = request.get("height", 1024)
        seed = request.get("seed")
        steps = request.get("steps", 20)
        denoise = request.get("denoise", 1)
        cfg = request.get("cfg", 8)
        sampler_name = request.get("sampler_name", "euler")
        tensors_file_name = request.get("tensors_file_name", None)
        payload_to_send = {
            "prompt_file": prompt_file,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "denoise": denoise,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "tensors_file_name": tensors_file_name,
            "image_input": image_input,
        }
        logger.info("Payload to send", payload_to_send)


        response = invoke_from_prompt(
            prompt_file=prompt_file,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            denoise=denoise,
            cfg=cfg,
            sampler_name=sampler_name,
            tensors_file_name=tensors_file_name,
            image_input=image_input,
        )
    except KeyError as e:
        logger.error(f"Error: {e}")
        return {
            "statusCode": 400,
            "body": json.dumps(
                {
                    "error": "Missing required parameter",
                }
            ),
        }

    image_data = response["Body"].read()

    result = {
        "headers": {"Content-Type": response["ContentType"]},
        "statusCode": response["ResponseMetadata"]["HTTPStatusCode"],
        "body": base64.b64encode(io.BytesIO(image_data).getvalue()).decode("utf-8"),
        "isBase64Encoded": True,
    }
    return result


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    event = {
        "body": "{\"positive_prompt\": \"hill happy dog\",\"negative_prompt\": \"hill\",\"prompt_file\": \"workflow_api.json\",\"seed\": 123}"
    }
    lambda_handler(event, None)
