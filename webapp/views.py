import os
from flask import render_template, request
from webapp import app

# 深層学習結果をロードする
from image2text_torch import Image2Text

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

estimator = Image2Text()

@app.route('/', methods=['GET'])
def show_main_page():
    return render_template("viewer.html")

@app.route('/estimate', methods=['POST'])
def estimate_category_from_image():
    save_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'upload.png')
    
    if 'image-file' in request.files:
        request.files['image-file'].save(save_filename)

    name = estimator.predict(save_filename)
    return name
