import gradio as gr
from gradio import Audio, Blocks, Button, Checkbox, Column, Dropdown, Examples, File, Group, Image, Row, Slider, Tab, Tabs, Textbox, Video

from matio.ios import public
from matio.ios.path import BANNER
from matio.page import InputArguments, OutputArguments, HyperParameterArguments

class Index(Blocks):
    def __init__(self,
                 title="Matio",
                 theme=gr.themes.Ocean(primary_hue="violet",
                                secondary_hue="rose",
                                font=[gr.themes.GoogleFont('Cascadia Code'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
                                font_mono=[gr.themes.GoogleFont('Cascadia Mono'), 'ui-monospace', 'Consolas', 'monospace']),
                 input_args=InputArguments(),
                 hyp_args=HyperParameterArguments(),
                 output_args=OutputArguments()):
        css = public.load("index.css")
        html = public.load("index.html")
        super().__init__(title=title,
                         css=css,
                         head=html,
                         theme=theme,
                         fill_height=False,
                         fill_width=False,
                         delete_cache=None,
                         analytics_enabled=False)
        with self:
            with Column(elem_classes="container"):
                with Group():
                    with Row(equal_height=True,height=400):
                        with Column(scale=4):
                            text_input = Textbox(visible=input_args.text)
                            image_input = Image(visible=input_args.image)
                            video_input = Video(visible=input_args.video)
                            audio_input = Audio(visible=input_args.audio)
                            file_input = File(visible=input_args.file)
                        with Column():
                            device = Dropdown(label="Device", choices=["Auto"], value="Auto", interactive=False)
                            seed = Textbox(label="Random Seed", value=hyp_args.seed)
                            generate = Button("🎯 Generate", variant="primary")

                with Row(equal_height=True):
                    with Tabs():
                        with Tab("Sampling"):
                            do_sample = Checkbox(
                                label="Enable Sampling",
                                value=True if hyp_args.do_sample is None else hyp_args.do_sample,
                                interactive=hyp_args.do_sample is None
                            )
                            temperature = Slider(
                                minimum=0.0, maximum=2.0, step=0.01,
                                value=0.7 if hyp_args.temperature is None else hyp_args.temperature,
                                label="temperature",
                                interactive=hyp_args.temperature is None
                            )
                            top_k = Slider(
                                minimum=0, maximum=100, step=1,
                                value=50 if hyp_args.top_k is None else hyp_args.top_k,
                                label="top_k",
                                interactive=hyp_args.top_k is None
                            )
                            top_p = Slider(
                                minimum=0.0, maximum=1.0, step=0.01,
                                value=0.9 if hyp_args.top_p is None else hyp_args.top_p,
                                label="top_p",
                                interactive=hyp_args.top_p is None
                            )

                        with Tab("Beam Search"):
                            do_beam_search = Checkbox(
                                label="Enable Beam Search",
                                value=False if hyp_args.do_beam_search is None else hyp_args.do_beam_search,
                                interactive=hyp_args.do_beam_search is None
                            )
                            early_stopping = Dropdown(
                                choices=["True", "False"],
                                value="True" if hyp_args.early_stopping is None else str(hyp_args.early_stopping),
                                label="early_stopping",
                                interactive=hyp_args.early_stopping is None
                            )
                            num_beams = Slider(
                                minimum=1, maximum=16, step=1,
                                value=4 if hyp_args.num_beams is None else hyp_args.num_beams,
                                label="num_beams",
                                interactive=hyp_args.num_beams is None
                            )
                            length_penalty = Slider(
                                minimum=0.1, maximum=5.0, step=0.1,
                                value=1.0 if hyp_args.length_penalty is None else hyp_args.length_penalty,
                                label="length_penalty",
                                interactive=hyp_args.length_penalty is None
                            )
                        
                        with Tab("Length Control"):
                            min_new_tokens = Slider(
                                minimum=1, maximum=1024, step=1,
                                value=128 if hyp_args.min_new_tokens is None else hyp_args.min_new_tokens,
                                label="min_new_tokens",
                                interactive=hyp_args.min_new_tokens is None
                            )
                            max_new_tokens = Slider(
                                minimum=1, maximum=1024, step=1,
                                value=128 if hyp_args.max_new_tokens is None else hyp_args.max_new_tokens,
                                label="max_new_tokens",
                                interactive=hyp_args.max_new_tokens is None
                            )
                            repetition_penalty = Slider(
                                minimum=0.1, maximum=5.0, step=0.1,
                                value=1.0 if hyp_args.repetition_penalty is None else hyp_args.repetition_penalty,
                                label="repetition_penalty",
                                interactive=hyp_args.repetition_penalty is None
                            )
                            no_repeat_ngram_size = Slider(
                                minimum=0, maximum=10, step=1,
                                value=3 if hyp_args.no_repeat_ngram_size is None else hyp_args.no_repeat_ngram_size,
                                label="no_repeat_ngram_size",
                                interactive=hyp_args.no_repeat_ngram_size is None
                            )

                    with Tabs():
                        with Tab(" "):
                            with Column(scale=4):
                                text_output = Textbox(visible=input_args.text)
                                image_output = Image(visible=input_args.image)
                                video_output = Video(visible=input_args.video)
                                audio_output = Audio(visible=input_args.audio)
                                file_output = File(visible=input_args.file)

            generate.click(
                fn=self.generate_fn,
                inputs=[text_input,
                        image_input,
                        video_input,
                        audio_input,
                        file_input,
                        seed,
                        do_sample,
                        temperature,
                        top_k,
                        top_p,
                        min_new_tokens,
                        max_new_tokens,
                        repetition_penalty,
                        no_repeat_ngram_size,
                        do_beam_search,
                        early_stopping,
                        num_beams,
                        length_penalty],
                outputs=[text_output,
                         image_output,
                         video_output,
                         audio_output,
                         file_output])

    def launch(self, inline = None, inbrowser = False, share = True, debug = False, max_threads = 40, auth = None, auth_message = None, prevent_thread_lock = False, show_error = False, server_name = "0.0.0.0", server_port = None, *, height = 500, width = "100%", favicon_path = BANNER, ssl_keyfile = None, ssl_certfile = None, ssl_keyfile_password = None, ssl_verify = True, quiet = False, show_api = True, allowed_paths = None, blocked_paths = None, root_path = None, app_kwargs = None, state_session_capacity = 10000, share_server_address = None, share_server_protocol = None, share_server_tls_certificate = None, auth_dependency = None, max_file_size = None, enable_monitoring = None, strict_cors = True, node_server_name = None, node_port = None, ssr_mode = None, pwa = None, mcp_server = None, _frontend = True, i18n = None):
        return super().launch(inline, inbrowser, share, debug, max_threads, auth, auth_message, prevent_thread_lock, show_error, server_name, server_port, height=height, width=width, favicon_path=favicon_path, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile, ssl_keyfile_password=ssl_keyfile_password, ssl_verify=ssl_verify, quiet=quiet, show_api=show_api, allowed_paths=allowed_paths, blocked_paths=blocked_paths, root_path=root_path, app_kwargs=app_kwargs, state_session_capacity=state_session_capacity, share_server_address=share_server_address, share_server_protocol=share_server_protocol, share_server_tls_certificate=share_server_tls_certificate, auth_dependency=auth_dependency, max_file_size=max_file_size, enable_monitoring=enable_monitoring, strict_cors=strict_cors, node_server_name=node_server_name, node_port=node_port, ssr_mode=ssr_mode, pwa=pwa, mcp_server=mcp_server, _frontend=_frontend, i18n=i18n)

    def generate_fn(self,
                    text,
                    image,
                    video,
                    audio,
                    file,
                    seed,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    min_new_tokens,
                    max_new_tokens,
                    repetition_penalty,
                    no_repeat_ngram_size,
                    do_beam_search,
                    early_stopping,
                    num_beams,
                    length_penalty):
        inputs = dict(
            text=text,
            image=image,
            video=video,
            audio=audio,
            file=file,
            seed=seed,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_beam_search=do_beam_search,
            early_stopping=early_stopping,
            num_beams=num_beams,
            length_penalty=length_penalty
        )
        hyps = dict(
            seed=seed,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_beam_search=do_beam_search,
            early_stopping=early_stopping,
            num_beams=num_beams,
            length_penalty=length_penalty
        )
        outputs = self.generate(inputs, hyps)
        assert isinstance(outputs, dict)
        return outputs.get("text",None), outputs.get("image",None), outputs.get("video",None), outputs.get("audio",None), outputs.get("file",None)