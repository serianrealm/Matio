import os
import importlib
from pathlib import Path

import gradio as gr
from gradio import Blocks, Button, Column, Dropdown, Group, Info, JSON, Markdown, Row, TabItem, Tabs, Textbox, Video

from matio.backend import Argument
from matio.backend import BaseModel

PUBLIC = Path(__file__).resolve().parent / "public"
MODELS = Path(__file__).resolve().parent.parent.parent / "models"
INTERFACES = Path(__file__).resolve().parent.parent.parent / "models"

def read(file):
        """
        Read CSS or HTML file, returns string
        
        Args:
            file (str): filepath
        
        Returns:
            str: content as Python string
        """
        filepath = PUBLIC / file

        if not os.path.exists(filepath):
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    
def scanfolder(folder:str):
        """
        递归扫描 ./models/interfaces 下的 python 文件，
        返回相对路径（去掉初始扫描目录和后缀）
        
        Args:
            folder (str): 相对于 self.MODELS 的子目录
        
        Returns:
            list[str]: 例如 ["foo/bar/modelA", "foo/modelB"]
        """
        base_dir = MODELS / folder
        return [path.relative_to(base_dir) for path in base_dir if os.path.isdir(path)]
    
def scanfile(file:str):
        """
        递归扫描 ./models/interfaces 下的 python 文件，
        返回相对路径（去掉初始扫描目录和后缀）
        
        Args:
            folder (str): 相对于 self.MODELS 的子目录
        
        Returns:
            list[str]: 例如 ["foo/bar/modelA", "foo/modelB"]
        """
        base_dir = MODELS / file
        return [path.relative_to(base_dir).with_suffix("") for path in base_dir if os.path.isdir(path)]


class Index(Blocks):
    def __init__(self, title = "Matio", **kwargs):
        css = self.read("index.css")
        html = self.read("index.html")
        theme=gr.themes.Ocean(primary_hue="violet",
                                secondary_hue="rose",
                                font=[gr.themes.GoogleFont('Cascadia Code'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
                                font_mono=[gr.themes.GoogleFont('Cascadia Mono'), 'ui-monospace', 'Consolas', 'monospace'])
        super().__init__(title=title,
                         css=css,
                         head=html,
                         theme=theme,
                         fill_height=False,
                         fill_width=False,
                         delete_cache=None,
                         analytics_enabled=False,
                         **kwargs)
        args = Argument()
        self.model = BaseModel()
        with self:
            with Column(elem_classes="container"):
                
                with Column():
                    Markdown("# 🎥 InternHands - Powered by vLLM")
                    Markdown("Sign language translation within videos")

                with Row():
                    with Column():
                        Markdown("### Configuration")
                        interfaces = Dropdown(
                            choices=scanfile("interfaces"),
                            label="Model Interfaces",
                        )
                        pretrained = Dropdown(
                            choices=scanfolder("pretrained"),
                            label="Pretrained Checkpoint",
                        )
                        adapter = Dropdown(
                            choices=scanfolder("adapter") + ["None"],
                            value="None",
                            label="Adapter Checkpoint",
                        )
                        load = Button("🚀 Load Model", variant="primary")

                        with Row():
                            quantization = Dropdown(
                                choices=["Auto","fp16","bf16","int8","int4"],
                                value="Auto",
                                label="quantization",
                            )
                            device = Dropdown(
                                choices=["Auto"],
                                value="Auto",
                                label="Device"
                            )

                    with Column():
                        with Tabs("Inference Parameters"):
                            Markdown("### Hyper Parameters")
                            with TabItem("Sampling"):
                                args.do_sample = Dropdown(choices=["True", "False"], value="True", label="do_sample")
                                args.temperature = Textbox(label="temperature", placeholder="e.g., 0.7")
                                args.top_k = Textbox(label="top_k", placeholder="e.g., 50")
                                args.top_p = Textbox(label="top_p", placeholder="e.g., 0.9")

                            with TabItem("Length Control"):
                                args.max_new_tokens = Textbox(label="max_new_tokens", placeholder="e.g., 128")
                                args.min_new_tokens = Textbox(label="min_new_tokens", placeholder="e.g., 10")
                                args.repetition_penalty = Textbox(label="repetition_penalty", placeholder="e.g., 1.2")
                                args.no_repeat_ngram_size = Textbox(label="no_repeat_ngram_size", placeholder="e.g., 3")

                            with TabItem("Beam Search"):
                                args.early_stopping = Dropdown(choices=["True", "False"], value="True", label="early_stopping")
                                args.num_beams = Textbox(label="num_beams", placeholder="e.g., 4")
                                args.length_penalty = Textbox(label="length_penalty", placeholder="e.g., 1.0")

                        configure = Button("Configure", variant="secondary")

                with Group():
                    Markdown("### Prompt")
                    with Column():
                        with Row():
                            video = Video(label="🎬 Video")
                            text = Textbox(label="Text")
                        generate = Button("🎯 Generate", variant="primary")
                
                with Group():
                    output = Textbox(label="Output")

            load.click(
                fn=self.load,
                inputs=[interfaces, pretrained, adapter, quantization, device],
            )

            configure.click(
                fn=self.configure,
                inputs=[args]
            )

            generate.click(
                fn=self.generate,
                inputs=[{
                    "text": text,
                    "video": video
                }],
                outputs=[output]
            )

    def load(self, interfaces, pretrained, adapter, quantization, device):
        try:
            module=importlib.import_module(str(INTERFACES/interfaces))
        except ImportError:
            return
        try:
            if model:=getattr(module, "Model"):
                self.model = model(pretrained, adapter, quantization, device)
        except AttributeError:
            return

    def configure(self, args):
        pass

    def generate(self, inputs):
        pass


def main():
    index=Index("InternHands - Powered by vLLM")
    index.launch(
        share=True,
        server_name="0.0.0.0",
        show_error=True,
        favicon_path="🎥"
    )

if __name__ == "__main__":
    main()