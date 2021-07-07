import os
from flask import Flask, flash, request, redirect, url_for, jsonify
import cv2


UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_input():
    if request.method == 'POST':
        # check if the post request has the input part
        if not request.json:
            print("no json")
            return redirect(request.url)
        if 'input' not in request.json:
            flash('No input')
            return redirect(request.url)
        input = request.json['input']
        # if user does not select input
        if input == '':
            flash('No selected input')
            return jsonify({ "Item": "why won't you say something"} )
        else:
            # detect(input)
            print(input)
            response = "you did it"
            return jsonify({ "Item": input} )
    return '''
    <!doctype html>
    <title>API</title>
    <h1>API Running chatbot Successfully</h1>'''

# def detect(image):

# 	return res


if __name__ == "__main__":
    app.run("0.0.0.0", port=80)