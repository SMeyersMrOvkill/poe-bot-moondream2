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
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
    def process_image(self, question, image, url):
        time.sleep(1.7329)
        yield "\n\n![image](%s)\n\n---" % url
        yield "\n\n```text\n"
        img = self._dcw_model.encode_image(image)
        yield self._dcw_model.answer_question(img, question, self._dcw_tokenizer)
        yield "\n\n```\n\n## "
        question = "Write a brief ( 2 - 8 words ) title for the image."
        yield self._dcw_model.answer_question(img, question, self._dcw_tokenizer)
        return
    def load(self):
        """
        Constructor

        :param path: path to the model repo on HF Hub
        :param text_model: path/HF Hub URL to the text model
        :param mmproj: path/HF Hub URL to the mmproj
        :return: MoondreamModel instance
        """
        self._dcw_model = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", device_map={'': 'cuda'}, revision = "2024-05-08", trust_remote_code=True)
        self._dcw_tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", device_map={'': 'cuda'}, revision = "2024-05-08", trust_remote_code=True)
    def empty_image(self):
        return "*Without an image, all I can see is a blank, grey wall.*\n\n```text/answer\n" + self._dcw_model.answer_question(self._dcw_model.encode_image(PImage.new("RGB", (192,192), (127,127,127))), "Write a short pun about this blank wall.", self._dcw_tokenizer) + "\n\n```\n\n"
    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        self.load()
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
                img = PImage.open(f"tmp."+typ)
                f.close()
                img = img.resize((192,192))
                img.save("tmp.png")
                imgs.append(img)
                imgs.append(att.url)
                break
        if len(imgs) < 2 and len(request.query[-1].content) > 0:
            img = PImage.new("RGB", (192,192), (127,127,127))
            img.save("tmp.png")
            with open("tmp.png", "rb") as f:
                imgs = [img, "data:img/png;base64," + base64.b64encode(f.read()).decode("utf-8")]
            res = self.process_image(request.query[-1].content, imgs[0], imgs[1])
            for tokn in res:
                yield fp.PartialResponse(text=tokn)
            return
        # Otherwise, process the image
        print(request.query[-1].content)
        txt = self.process_image(request.query[-1].content, imgs[0], imgs[1])
        text = "\n\n---\n\n"
        for x in txt:
            print(x)
            for y in x:
                print(y)
                for z in y:
                    print(z)
                    tokn = z
                    text += tokn
                    yield fp.PartialResponse(text=tokn)
        print(text)
        yield fp.PartialResponse(text=text, is_replace_response=True)
        return

REQUIREMENTS = ["fastapi-poe==0.0.28", "requests", "transformers", "einops", "huggingface-hub", "accelerate", "pillow", "torch", "torchvision"]
image = Image.debian_slim().pip_install(*REQUIREMENTS)
app = Stub("dcw-moondream2")
voldemort = modal.Volume.from_name("voldemort", create_if_missing=True)

@app.function(image=image, gpu=modal.gpu.T4(), volumes={"/voldemort": voldemort}, keep_warm=1, enable_memory_snapshot=True)
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