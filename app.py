from ObjectDetector import Detector
import io
from flask import Flask, render_template, request, send_from_directory, send_file, jsonify
from PIL import Image
import requests
import os
import img_transforms
from flask_cors import CORS, cross_origin
import logging
import base64
import numpy as np

# from OpenSSL import SSL
# context = SSL.Context(SSL.PROTOCOL_TLSv1_2)
# context.use_privatekey_file('server.key')
# context.use_certificate_file('server.crt')   

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
detector = Detector()

RENDER_FACTOR = 5


# function to load img from url
def load_image_url(url):
	response = requests.get(url)
	img = Image.open(io.BytesIO(response.content))
	return img

# run inference using image transform to reduce memory
def run_inference_transform(img_path = 'file.jpg', transformed_path = 'file_transformed.jpg'):

	# get height, width of image
	original_img = Image.open(img_path)

	# transform to square, using render factor
	transformed_img = img_transforms._scale_to_square(original_img, targ=RENDER_FACTOR*16)
	transformed_img.save(transformed_path)

	# run inference using detectron2
	untransformed_result, _ = detector.inference(transformed_path)

	# unsquare
	result_img = img_transforms._unsquare(untransformed_result, original_img)

	# clean up
	try:
		os.remove(img_path)
		os.remove(transformed_path)
	except:
		pass

	return result_img

# run inference using detectron2
def run_inference(img_path = 'file.jpg'):

	# run inference using detectron2
	result_img, cnts_data = detector.inference(img_path)

	# clean up
	try:
		os.remove(img_path)
	except:
		pass

	return result_img, cnts_data


# run inference using detectron2
def run_inference_im(img):

	# run inference using detectron2
	result_img, cnts_data = detector.inference_im(img)

	return result_img, cnts_data

@app.route("/")
def index():
	return render_template('index.html')


@app.route("/detect", methods=['POST', 'GET'])
@cross_origin()
def upload():
	content = request.json
	imagebase64 = content['image']
	rgb_im = None
	cnts_data = None
	if request.method == 'POST':
		try:
			imagebase64 = imagebase64.split(";")[1].split(",")[1]
			file = Image.open(io.BytesIO(base64.b64decode(imagebase64)))
			# remove alpha channel
			rgb_im = file.convert('RGB')
			rgb_im.save('file.jpg')


		# failure
		except:

			return render_template("failure.html")

	if rgb_im:
		# result_img, cnts_data = run_inference_im(rgb_im)
		result_img, cnts_data = run_inference('file.jpg')
	else:
		return render_template("failure.html")

	# create file-object in memory
	file_object = io.BytesIO()

	# write PNG in file-object
	result_img.save(file_object, 'PNG')

	# move to beginning of file so `send_file()` it will read from start    
	file_object.seek(0)

	a = base64.b64encode(file_object.read())
	a = a.decode('utf-8')
	# print(a)

	# assume bytes_io is a `BytesIO` object
	# byte_str = file_object.read()

	# Convert to a "unicode" object
	# text_obj = byte_str.decode('UTF-8')  # Or use the encoding you expect
	# print(text_obj)
	# print(cnts_data)
	# cnts_data = np.concatenate(cnts_data, axis=0)
	# print(cnts_data.shape)
	data = {
		"image": a,
		"cnts": cnts_data
	}

	# r = jsonify(data)
	# print(r)
	# print('response is, only sending an image back for now', r)
	return data

	return send_file(file_object, mimetype='image/jpeg')


if __name__ == "__main__":
	logging.getLogger('flask_cors').level = logging.DEBUG
	# get port. Default to 8080
	port = int(os.environ.get('PORT', 8080))

	# run app
	app.run(host='0.0.0.0', port=port,ssl_context=("cert.pem", "key.pem"))


