from flask import Flask, request, make_response, Response, send_file
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage

import numpy as np
import cv2
import os

from detector import Detector

app = Flask(__name__)
api = Api(app)

@app.route("/")
def home():
    return "Hello! This is a rest-api for object detection."

class ObjectDetect(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument('image', type=FileStorage, help='The image to be used for object detection.')
    
    IMAGE_ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

    def media_file_allowed(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.IMAGE_ALLOWED_EXTENSIONS

    def post(self):

        img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        print(img.shape)
        
        # Perform detection.
        detector = Detector()

        img_detect = detector.detect_and_draw(img)
        # print(type(img))
        if type(img_detect) == int:
            cv2.imwrite('image.jpg', img)
        else:
            cv2.imwrite('image.jpg', img_detect)
            # return 'No objects detected.'

        return 'Successful', 200
        # return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')

    def get(self):
        return send_file('image.jpg', mimetype='image/get')

api.add_resource(ObjectDetect, '/detect')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=33, debug=True)