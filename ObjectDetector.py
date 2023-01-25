import cv2 as cv
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
import torch
import numpy as np
from PIL import Image
from functools import reduce 
import logging
from sys import stdout

class Detector:

	def __init__(self):

		# set model and test set
		self.model = 'mask_rcnn_R_50_FPN_3x.yaml'

		# obtain detectron2's default config
		self.cfg = get_cfg() 

		# load values from a file
		# self.cfg.merge_from_file("test.yaml")
		self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/"+self.model)) 

		# set device to cpu
		self.cfg.MODEL.DEVICE = "cpu"

		# get weights 
		self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/"+self.model) 
		# self.cfg.MODEL.WEIGHTS = "model_final_f10217.pkl"

		# set the testing threshold for this model
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  

		# build model from weights
		# self.cfg.MODEL.WEIGHTS = self.convert_model_for_inference()

	# build model and convert for inference
	def convert_model_for_inference(self):

		# build model
		model = build_model(self.cfg)

		# save as checkpoint
		torch.save(model.state_dict(), 'checkpoint.pth')

		# return path to inference model
		return 'checkpoint.pth'

	# detectron model
	# adapted from detectron2 colab notebook: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5	
	def inference(self, file):

		predictor = DefaultPredictor(self.cfg)
		im = cv.imread(file)
		outputs = predictor(im)

		# with open(self.curr_dir+'/data.txt', 'w') as fp:
		# 	json.dump(outputs['instances'], fp)
		# 	# json.dump(cfg.dump(), fp)

		# get metadata
		metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

		# visualise
		v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

		# get image 
		img = Image.fromarray(np.uint8(v.get_image()[:, :, ::-1]))

		all_cnts = np.zeros_like(outputs["instances"].pred_masks)

		# all_cnts = np.expand_dims(all_cnts, axis=0)
		# # get contours for each mask
		all_cnts = []
		for idx, mask in enumerate(outputs["instances"].pred_masks):
			mask_img = mask.cpu().numpy()
			mask_img = mask_img.astype('uint8')
			cnts = cv.findContours(mask_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			all_cnts.append(cnts[0][0].tolist())

		# write to jpg
		# cv.imwrite('img.jpg',v.get_image())

		return img, all_cnts

	# detectron model
	# adapted from detectron2 colab notebook: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5	
	def inference_im(self, im):

		predictor = DefaultPredictor(self.cfg)
		outputs = predictor(im)

		# with open(self.curr_dir+'/data.txt', 'w') as fp:
		# 	json.dump(outputs['instances'], fp)
		# 	# json.dump(cfg.dump(), fp)

		# get metadata
		metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

		# visualise
		v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

		# get image 
		img = Image.fromarray(np.uint8(v.get_image()[:, :, ::-1]))

		all_cnts = np.zeros_like(outputs["instances"].pred_masks)
		# all_cnts = np.expand_dims(all_cnts, axis=0)
		# # get contours for each mask
		# for idx, mask in enumerate(outputs["instances"].pred_masks):
		# 	mask_img = mask.cpu().numpy()
		# 	mask_img = mask_img.astype('uint8')
		# 	cnts = cv.findContours(mask_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		# 	all_cnts[idx] = cnts

		# write to jpg
		# cv.imwrite('img.jpg',v.get_image())

		return img, all_cnts


