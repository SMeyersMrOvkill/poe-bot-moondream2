from __future__ import annotations

import base64
from typing import AsyncIterable

import fastapi_poe as fp
from modal import Image, Stub, asgi_app
import requests
import json
import ast
import os
import requests
from PIL import Image as PImage
import modal

from transformers import TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM
import torch

import threading

class MoondreamModel():
  def __init__(self, model: str, tokenizer: str):
    """
    Constructor

    :param path: path to the model repo on HF Hub
    :param text_model: path/HF Hub URL to the text model
    :param mmproj: path/HF Hub URL to the mmproj
    :return: MoondreamModel instance
    """
    self.model = AutoModelForCausalLM.from_pretrained(model, device_map={'': 'cuda'}, torch_dtype=torch.float16, revision = "2024-04-02", trust_remote_code=True)
    self.model.eval()
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, cache_dir="/voldemort", device_map={'': 'cuda'}, revision = "2024-04-02", trust_remote_code=True)
    self.device="cuda:0"
    self.streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
  
  def format(self, content):
    with open("/voldemort/blank.png", "rb") as f:
        return "*Without an image, all I can see is a blank, grey wall.*\n\n```text\n" + self.model.answer_question(self.model.encode_image(PImage.open("/voldemort/blank.png")), content, self.tokenizer, streamer=self.streamer) + "\n\n```\n\n"

  def _infer(self, **kwargs):
        img = kwargs['img']
        prompt = kwargs['prompt']
        img = self.model.encode_image(img)
        return self.model.answer_question(image_embeds=img, question=prompt, tokenizer=self.tokenizer, streamer=self.streamer)
  
  def __call__(self, message, **kwargs):
    """
    Overrides Object.__call__(self, **kwargs)

    Okay, this one takes a bit of explaining. Buckle in.
    *. messages; A list[dict[str,str]], which just means a list, in which each element is a pair of strings. *COUGH* I miss when tuples were enough *fixes cane*
    *. top_p; A humble float whose value is, considered simply, "How likely is it, statistically, that the next 3/4 of a word is likely to be said, or even more simply, "How much does what i'm about to say make sense?"
    *. top_k; An integer that acts like a bouncer at a club. It decides how many of the "most likely next words" get to come in and party (be considered for the next word in the sequence).
    *. temperature; The overarching "spice" or "chaos" factor. It controls the randomness of the next word selection, with a lower value making the model more deterministic (more likely to choose the highest-probability word), and a higher value making the model more random (more likely to choose less probable words).
    *. max_tokens; The maximum number of tokens to generate. Makes enough sense on it's own, THAT MAKES ONE!!!
    *. stop; The stop sequence(s) list of strings exists to tell a model when to quit. Because they *DON'T* have a clue.
    :param messages: list of messages
    :param top_p: The top percentage of tokens to consider
    :param top_k: The top number of tokens to consider
    :param temperature: The temperature of the model
    :param max_tokens: The maximum number of tokens to generate
    :param stop: The stop sequence(s), a list[str]
    :return: LLM response
    """
    img: PImage = None
    content = message['content']
    txt = None
    for cont in content:
        if cont['type'] == 'text':
            txt = cont['text']
        if cont['type'] == "image":
            img = cont['image']
    if txt is None and img is None:
        return self.format(".")
    if txt is None:
        txt = "Describe the image."
    if img is None:
        img = PImage.open("/voldemort/blank.png")
    t = threading.Thread(target=self._infer, kwargs={
        "img": img,
        "prompt": txt
    })
    t.run()
    yield self.streamer

MAX_TOKENS = 8192
MAX_INPUT = (MAX_TOKENS/2)
MAX_OUTPUT = (MAX_TOKENS/2)
SYSTEM_PROMPT = "You are a helpful, morally upstanding assistant. Follow directions to the letter. Pay attention."

def mkhist(history: list):
    shadowhist = []
    count = 0
    # Iterate backward through the OpenAI Chat History to trim the length and extract from obj to dict.
    for i in range(len(history) - 1, -1, -1):
        if count < 2:
            shadowhist.append(
                {
                    "role": history[i].role,
                    "content": [
                        {
                            "type": "text",
                            "text": history[i].content
                        }
                    ]
                }
            )
            count += 1
        else:
            break
    return shadowhist

def mkreq(history):
    req = {
        "max_tokens": MAX_OUTPUT,
        "temperature": 0.123,
        "top_p": 0.85,
        "top_k": 100,
        "streaming": True,
    }
    mdl_instance = MoondreamModel(model="/voldemort/moondream2", tokenizer="vikhyatk/moondream2")
    print(history)
    res = mdl_instance(message=history[-2], **req)
    return res

class Moondream2Bot(fp.PoeBot):
    def __init__(self):
        super().__init__()
    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        return fp.SettingsResponse(
            allow_attachments=True,
            insert_attachments_to_message=True,
            enable_image_comprehension=True,
            enable_multi_bot_chat_prompting=True
        )
    async def insert_attachment_messages(self, query_request: fp.QueryRequest) -> fp.QueryRequest:
        print(query_request.query.attachments)
    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        hist = mkhist(request.query)
        attachments = request.query[-1].attachments
        print(attachments)
        imgs = []
        for att in attachments:
            if att.content_type.startswith("image"):
                typ = "png"
                if att.content_type.endswith("jpeg") or att.content_type.endswith("jpg"):
                    typ = "jpeg"
                if att.content_type.endswith("gif"):
                    typ = "gif"
                res = requests.get(att.url)
                if res.status_code != 200:
                    print("Download error")
                bts = res.content
                with open("tmp." +typ, "wb") as f:
                    f.write(res.content)
                    f.flush()
                img = None
                with PImage.open("tmp."+typ) as img:
                    img = img.resize((256,256))
                    img.save("tmp."+typ)
                imgs.append(img)
        if len(hist) < 2:
            hist = [
                *hist,
                {
                    "role": 'user',
                    "content": [
                        {
                            "type": "text",
                            "text": "."
                        },
                        {
                            "type": "image",
                            "image": PImage.open("/voldemort/blank.png")
                        }
                    ]
                }
            ]
        if len(imgs) > 0:
            for img in imgs:
                hist[-2]['content'].append({
                    "type": "image",
                    "image": img
                })
                break
        txt = mkreq(hist)
        text = ""
        enumerate(txt)
        for x in txt:
            for y in x:
                text += y
                yield fp.PartialResponse(text=text)
            break
        print(text)
        yield fp.PartialResponse(text=text, is_replace_response=True)
        return

REQUIREMENTS = ["fastapi-poe==0.0.28", "requests", "transformers", "einops", "huggingface-hub", "accelerate", "pillow", "torch", "torchvision"]
image = Image.debian_slim().pip_install(*REQUIREMENTS)
app = Stub("dcw-moondream2")
voldemort = modal.Volume.from_name("voldemort")

@app.function(image=image, gpu=modal.gpu.T4(), volumes={"/voldemort": voldemort}, keep_warm=1)
@asgi_app()
def fastapi_app():
    bot = Moondream2Bot()
    # Optionally, provide your Poe access key here:
    # 1. You can go to https://poe.com/create_bot?server=1 to generate an access key.
    # 2. We strongly recommend using a key for a production bot to prevent abuse,
    # but the starter examples disable the key check for convenience.
    # 3. You can also store your access key on modal.com and retrieve it in this function
    # by following the instructions at: https://modal.com/docs/guide/secrets
    POE_ACCESS_KEY = "NhH6TpgRCKaRVpDGIwoULh2ZNwkwqVrU"
    # app = make_app(bot, access_key=POE_ACCESS_KEY)
    return fp.make_app(bot, access_key=POE_ACCESS_KEY)