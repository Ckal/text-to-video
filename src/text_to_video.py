import torch

from transformers.tools.base import Tool, get_default_device
from transformers.utils import is_accelerate_available

from diffusers import DiffusionPipeline


TEXT_TO_VIDEO_DESCRIPTION = (
    "This is a tool that creates a video according to a text description. It takes an input named `prompt` which "
    "contains the image description, as well as an optional input `seconds` which will be the duration of the video. "
    "The default is of two seconds. The tool outputs a video object."
)


class TextToVideoTool(Tool):
    default_checkpoint = "damo-vilab/text-to-video-ms-1.7b"
    description = TEXT_TO_VIDEO_DESCRIPTION
    inputs = ['text']
    outputs = ['video']

    def __init__(self, device=None, **hub_kwargs) -> None:
        if not is_accelerate_available():
            raise ImportError("Accelerate should be installed in order to use tools.")

        super().__init__()

        self.device = device
        self.pipeline = None
        self.hub_kwargs = hub_kwargs

    def setup(self):
        if self.device is None:
            self.device = get_default_device()

        self.pipeline = DiffusionPipeline.from_pretrained(
            self.default_checkpoint, variant="fp16"
        )
        self.pipeline.to(self.device)

        self.is_initialized = True

    def __call__(self, prompt, seconds=2):
        if not self.is_initialized:
            self.setup()

        return self.pipeline(prompt, num_frames=8 * seconds).frames

