# tools/image_agent.py
"""
Image Agent for generating social media images using OpenAI's DALL·E 3.
"""

import os
import uuid
import logging
from pathlib import Path
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
IMAGES_DIR = Path("data/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def create_image(prompt: str, channel: str) -> dict:
    """
    Generate an image for a social media post using DALL·E 3.
    Args:
        prompt: The image prompt.
        channel: The social media channel (e.g., 'instagram', 'x', 'linkedin').
    Returns:
        Dictionary with 'url' and 'path' of the generated image.
    Raises:
        Exception: If the API call fails or the image cannot be saved.
    """
    try:
        logger.info(f"Generating image with DALL·E 3 for channel: {channel}")
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="url"
        )
        logger.info(f"DALL·E 3 API response: {response}")

        image_url = response.data[0].url
        image_id = str(uuid.uuid4())
        image_path = IMAGES_DIR / f"{channel}_{image_id}.png"

        logger.info(f"Downloading image from {image_url}")
        image_response = requests.get(image_url, timeout=10)
        image_response.raise_for_status()
        with open(image_path, "wb") as f:
            f.write(image_response.content)

        return {"url": image_url, "path": str(image_path)}

    except Exception as e:
        raise Exception(f"Failed to generate or save image with DALL·E 3: {str(e)}")