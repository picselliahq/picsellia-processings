import argparse
import random
import re
import time
from difflib import get_close_matches

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import sys
import os

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append("{}/MiniGPT-4".format(script_directory))

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import (Chat, Conversation,
                                                SeparatorStyle)
from PIL import Image



def setup_parser(cfg_path="eval_configs/minigpt4_eval.yaml"):
    args = argparse.Namespace()
    args.cfg_path = cfg_path
    args.gpu_id = 0
    args.options = []
    return args


def setup(
    architecture="minigpt4",
    image_size=224,
    llama_model="Vision-CAIR/vicuna",
    num_query_token=32,
    ckpt="checkpoints/pretrained_minigpt4.pth",
):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cudnn.benchmark = False
    cudnn.deterministic = True

    print("Initializing Chat")
    if architecture == "minigpt4":
        args = setup_parser()
    elif architecture == "minigpt4_llama2":
        args = setup_parser(f"{script_directory}/MiniGPT-4/eval_configs/minigpt4_llama2_eval.yaml")
    elif architecture == "minigpt_v2":
        args = setup_parser(f"{script_directory}/MiniGPT-4/eval_configs/minigptv2_eval.yaml")
    
    cfg = Config(args)

    device = "cuda:{}".format(args.gpu_id)

    model_config = cfg.model_cfg
    print(f'model config: {model_config}')
    model_config.arch = architecture
    model_config.image_size = image_size
    model_config.llama_model = llama_model
    model_config.num_query_token = num_query_token
    model_config.ckpt = ckpt
    model_config.low_resource = False
    
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
        vis_processor_cfg
    )

    model = model.eval()
    return model, vis_processor, device


def get_conversation(architecture="minigpt4"):
    if architecture == "minigpt4":
        CONV_VISION = Conversation(
            system="Give the following image: <Img>ImageContent</Img>. "
            "You will be able to see the image once I provide it to you. Please answer my questions.",
            roles=("Human: ", "Assistant: "),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="###",
        )
    elif architecture == "minigpt4_llama2":
        CONV_VISION = Conversation(
            system="Give the following image: <Img>ImageContent</Img>. "
            "You will be able to see the image once I provide it to you. Please answer my questions.",
            roles=("<s>[INST] ", " [/INST] "),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )
    elif architecture == "minigpt_v2":
        CONV_VISION = Conversation(
            system="",
            roles=("<s>[INST] ", " [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )
    return CONV_VISION


def resize_image(image_path, size):
    image = Image.open(image_path)
    resized_image = image.resize((size, size))
    return resized_image


def extract_substrings(string):
    index = string.rfind("}")
    if index != -1:
        string = string[: index + 1]

    pattern = r"<p>(.*?)\}(?!<)"
    matches = re.findall(pattern, string)
    substrings = [match for match in matches]

    return substrings


def find_more_similar_label(llm_label, label_matching):
    close_match = get_close_matches(llm_label, label_matching.keys(), n=1)
    print(f"Close match: {close_match}")
    if close_match:
        return close_match[0]
    else:
        return list(label_matching.keys())[0]


class MiniGPT4Predictor:
    def __init__(
        self, model_architecture="minigpt_v2", llama_model="meta-llama/Llama-2-7b-chat-hf", image_size=448, num_query_token=32, checkpoint_path=None
    ):
        self.model, self.vis_processor, self.device = setup(
            architecture=model_architecture,
            llama_model=llama_model,
            image_size=image_size,
            num_query_token=num_query_token,
            ckpt=checkpoint_path,
        )
        self.conv = get_conversation(model_architecture)
        self.image_size = image_size
        self.num_query_token = num_query_token
        self.conv = get_conversation(model_architecture)
        self.chat = Chat(self.model, self.vis_processor, device=self.device)
        self.inference_time = []

    def _upload_img(self, image_path, chat_state):
        img_list = []
        resized_image = resize_image(image_path, self.image_size)
        _ = self.chat.upload_img(resized_image, chat_state, img_list)
        self.chat.encode_img(img_list)
        return img_list

    def predict(
        self, image_path, prompt, temperature=0.2, max_new_tokens=500, max_length=2000
    ):
        tic = time.perf_counter()
        chat_state = self.conv.copy()
        img_list = self._upload_img(image_path, chat_state)
        self.chat.ask(prompt, chat_state)
        llm_message = self.chat.answer(
            conv=chat_state,
            img_list=img_list,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
        )[0]
        toc = time.perf_counter()
        self.inference_time.append(toc - tic)
        return llm_message

    def get_evaluated_labels(self, image_path, prompt, label_matching):
        llm_answer = self.predict(image_path, prompt)
        closest_label = find_more_similar_label(llm_answer, label_matching)
        confidence = 1.0
        evaluated_labels = [(label_matching[closest_label], confidence)]
        return evaluated_labels

    def get_tags(self, image_path, prompt, label_matching):
        llm_answer = self.predict(image_path, prompt)
        print(f"LLM answer: {llm_answer}")
        closest_label = find_more_similar_label(llm_answer, label_matching)
        return label_matching[closest_label]