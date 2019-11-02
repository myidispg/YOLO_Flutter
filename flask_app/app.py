from flask import Flask, request, make_response
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage

import numpy as np
import cv2

app = Flask(__name__)
api = Api(app)

@app.route("/")
def home():
    return "Hello, Flask!"

class ObjectDetect(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument('image', type=FileStorage, help='The image to be used for object detection.')
    
    IMAGE_ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

    def media_file_allowed(self, filename, media_type):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.IMAGE_ALLOWED_EXTENSIONS

    def post(self):

        img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        print(img.shape)

        # Encode the image back to jpeg
        _, buffer = cv2.imencode('.jpg', img)
        response = make_response(buffer.tobytes())
        print(type(response))
        return response

api.add_resource(ObjectDetect, '/detect')

if __name__ == '__main__':
    app.run(debug=True)