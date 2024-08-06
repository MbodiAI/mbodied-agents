import json
from typing import List

import numpy as np

from mbodied.agents.sense.sensory_agent import SensoryAgent
from mbodied.types.sense.scene import BBox2D, PixelCoords
from mbodied.types.sense.vision import Image


class SegmentationAgent(SensoryAgent):
    """An image segmentation agent that uses a remote segmentation server to segment objects in an image."""

    def __init__(
        self,
        model_src="https://api.mbodi.ai/sense/",
        model_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            model_src=model_src,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    def act(
        self,
        image: Image,
        input_data: BBox2D | List[BBox2D] | PixelCoords,
        *args,
        api_name: str = "/segment",
        **kwargs,
    ) -> tuple[Image, np.ndarray]:
        """Perform image segmentation using the remote segmentation server.

        Args:
            image (Image): The image to act on.
            input_data (Union[BBox2D, List[BBox2D], PixelCoords]): The input data for segmentation, either a bounding box,
                a list of bounding boxes, or pixel coordinates.
            *args: Variable length argument list.
            api_name (str): The name of the API endpoint to use.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tuple[Image, np.ndarray]: The segmented image and the masks of the segmented objects.
        """
        if self.actor is None:
            raise ValueError("Remote actor for agent not initialized.")

        if isinstance(input_data, PixelCoords):
            input_type = "Coordinates"
            input_data_str = f"{input_data.x},{input_data.y}"
        elif isinstance(input_data, BBox2D):
            input_type = "Bounding Boxes"
            input_data_str = json.dumps([[input_data.x1, input_data.y1, input_data.x2, input_data.y2]])
        elif isinstance(input_data, list) and all(isinstance(box, BBox2D) for box in input_data):
            input_type = "Bounding Boxes"
            input_data_str = json.dumps([[box.x1, box.y1, box.x2, box.y2] for box in input_data])
        else:
            raise ValueError("Unsupported input type. Must be BBox2D, List[BBox2D], or PixelCoords.")

        segmented_image, masks = self.actor.predict(
            image.base64, input_type, input_data_str, *args, api_name=api_name, **kwargs
        )
        # Convert gradio Dataframe numpy to numpy array.
        masks = np.array(masks["data"])
        return Image(segmented_image), masks


# Exmaple usage:
if __name__ == "__main__":
    agent = SegmentationAgent(model_src="https://api.mbodi.ai/sense/")
    bboxes = [BBox2D(x1=225, y1=196, x2=408, y2=355), BBox2D(x1=378, y1=179, x2=494, y2=236)]
    mask_image, masks = agent.act(image=Image("resources/bridge_example.jpeg"), input_data=bboxes)
    print("Masks shape", masks.shape)
    mask_image.pil.show()

    pixel_coords = PixelCoords(x=800, y=100)
    mask_image, masks = agent.act(image=Image("resources/bridge_example.jpeg"), input_data=pixel_coords)
    print("Masks shape", masks.shape)
    mask_image.pil.show()
