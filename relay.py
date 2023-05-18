import requests
from PIL import Image
from gradio.processing_utils import encode_pil_to_base64
import base64
import json

import io, base64
from PIL import Image


# url_img2img = "http://192.168.1.36:7860/sdapi/v1/img2img"
# simple_img2img = {
#     "init_images": [encode_pil_to_base64(Image.open(r"/home/m/Downloads/PXL_20221231_155347511.jpg"))],
#     "resize_mode": 0,
#     "denoising_strength": 0.5,
#     "mask": None,
#     "mask_blur": 4,
#     "inpainting_fill": 0,
#     "inpaint_full_res": True,
#     "inpaint_full_res_padding": 0,
#     "inpainting_mask_invert": 0,
#     "prompt": "golden hour",
#     "styles": [],
#     "seed": -1,
#     "subseed": -1,
#     "subseed_strength": 0,
#     "seed_resize_from_h": -1,
#     "seed_resize_from_w": -1,
#     "batch_size": 1,
#     "n_iter": 1,
#     "steps": 80,
#     "cfg_scale": 7,
#     "width": 512,
#     "height": 512,
#     "restore_faces": False,
#     "tiling": False,
#     "negative_prompt": "",
#     "eta": 0,
#     "s_churn": 0,
#     "s_tmax": 0,
#     "s_tmin": 0,
#     "s_noise": 1,
#     "override_settings": {},
#     "sampler_index": "Euler a",
#     "include_init_images": False
#     }

url_img2img = "http://192.168.1.36:7860/sdapi/v1/txt2img"
simple_img2img = {
    "enable_hr": False,
    "denoising_strength": 0,
    "firstphase_width": 0,
    "firstphase_height": 0,
    "prompt": "a cat jumping over a dog in a backyard, animals, fighting, cat dog",
    "styles": [],
    "seed": -1,
    "subseed": -1,
    "subseed_strength": 0,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "batch_size": 1,
    "n_iter": 1,
    "steps": 80,
    "cfg_scale": 8,
    "width": 512,
    "height": 512,
    "restore_faces": False,
    "tiling": False,
    "negative_prompt": "",
    "eta": 0,
    "s_churn": 0,
    "s_tmax": 0,
    "s_tmin": 0,
    "s_noise": 1,
    "sampler_index": "Euler a"
}

x = requests.post(url_img2img, json = simple_img2img)
json_reponse = json.loads(x.text)
print(json_reponse)
img_data = json_reponse['images'][0]
print(img_data)
# with open("imageToSave.png", "wb") as fh:
#     fh.write(base64.decodebytes(img_data))

# Assuming base64_str is the string value without 'data:image/jpeg;base64,'
img = Image.open(io.BytesIO(base64.decodebytes(bytes(img_data, "utf-8"))))
img.save('my-image.jpeg')
