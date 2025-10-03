from matio import Index, InputArguments, OutputArguments, HyperParameterArguments
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import PeftModel
from concurrent.futures import ThreadPoolExecutor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(image_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose([T.ToImage(),
                           T.ToDtype(torch.float32, scale=True),
                           T.Resize((image_size, image_size),interpolation=InterpolationMode.BICUBIC),
                           T.Normalize(MEAN, STD)])

def load_video(video_path):
    num_segments = 128
    model_config = AutoConfig.from_pretrained("OpenGVLab/InternVideo2_5_Chat_8B",
                                              trust_remote_code=True)
    vr = VideoReader(video_path)
    transform = build_transform(image_size=model_config.force_image_size or model_config.vision_config.image_size)
    num_frames, fps = len(vr), float(vr.get_avg_fps())
    indices = np.linspace(0, num_frames-1, num_segments, dtype=int)
    images = vr.get_batch(indices).asnumpy()
    with ThreadPoolExecutor(max_workers=8) as executor:
        pixel_values = torch.stack(list(executor.map(transform, images)))
    return pixel_values, num_frames

class ModelIndex(Index):
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVideo2_5_Chat_8B", trust_remote_code=True)
    model = AutoModel.from_pretrained("OpenGVLab/InternVideo2_5_Chat_8B", trust_remote_code=True).half().cuda().to(torch.bfloat16)
    model = PeftModel.from_pretrained(model, "models/checkpoint-3000")
    generation_config = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=1024,
        top_p=0.1,
        num_beams=1
    )
    def generate(self, inputs, **kwargs):
        with torch.inference_mode():
            pixel_values, num_frames = load_video(video_path=inputs["video"])
            pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_frames))])
            question = "This is a sign language conversation and you are the translator. Pay attention to the person at the center, and pay attention to his or her actions and movements. Try to understand sign language within frames, and translate into natural language that appears in chat in common daily life. Instead of translating the imaginary content, only translate the content you see that can determine confidently. Do not describe the content, but only output what the person says in natural language."
            prompt = video_prefix + question
            output = self.model.chat(self.tokenizer, pixel_values, prompt, self.generation_config, return_history=False)
        print(output)
        return output
            

def main():
    inputs = InputArguments(video=True)
    hyps = HyperParameterArguments(do_sample=False,
                                   temperature=0.0,
                                   max_new_tokens=1024,
                                   top_p=0.1,
                                   num_beams=1)
    outputs = OutputArguments(text=True)
    index = ModelIndex(input_args=inputs, hyp_args=hyps, output_args=outputs)
    index.launch()

if __name__ == "__main__":
    main()



