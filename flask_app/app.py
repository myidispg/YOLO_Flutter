from flask import Flask, request, make_response, Response
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

        img = detector.detect_and_draw(img)
        # print(type(img))
        if type(img) == int:
            return 'No objects detected.'


        # Encode the image back to jpeg
        _, jpg = cv2.imencode('.jpg', img)
        response = make_response(jpg.tobytes())
        print(type(response))
        return response
        # return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')

api.add_resource(ObjectDetect, '/detect')

if __name__ == '__main__':
    app.run(debug=True)