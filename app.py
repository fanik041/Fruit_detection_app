# app.py
import warnings
from keras.models import load_model
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    def no_warn():
        def fxn():
            warnings.warn("deprecated", DeprecationWarning)

            with warnings.catch_warnings(record=True) as w:
                # Cause all warnings to always be triggered.
                warnings.simplefilter("always")
                # Trigger a warning.
                fxn()
                # Verify some things
                assert len(w) == 1
                assert issubclass(w[-1].category, DeprecationWarning)
                assert "deprecated" in str(w[-1].message)

    model = load_model('Fruits_360.h5')
    no_warn()

    print(model.summary())

    class Fruit:

        def __init__(self, img_dir=''):
            self.img_dir = img_dir
            self.cnt = 0
            self.batch_holder = None
            self.model = load_model('Fruits_360.h5')
            self.Label_dict = labels = {'Apple Braeburn': 0,
                                        'Apple Golden': 1,
                                        'Apple Granny Smith': 2,
                                        'Apple Red': 3,
                                        'Apricot': 4,
                                        'Avocado': 5,
                                        'Avocado ripe': 6,
                                        'Banana': 7,
                                        'Banana Lady Finger': 8,
                                        'Banana Red': 9,
                                        'Cactus fruit': 10,
                                        'Cantaloupe 1': 11,
                                        'Cantaloupe 2': 12,
                                        'Carambula': 13,
                                        'Cherry 1': 14,
                                        'Cherry Wax Black': 15,
                                        'Cherry Wax Red': 16,
                                        'Cherry Wax Yellow': 17,
                                        'Chestnut': 18,
                                        'Clementine': 19,
                                        'Cocos': 20,
                                        'Dates': 21,
                                        'Grape Blue': 22,
                                        'Grape Pink': 23,
                                        'Grape White': 24,
                                        'Grapefruit Pink': 25,
                                        'Grapefruit White': 26,
                                        'Guava': 27,
                                        'Hazelnut': 28,
                                        'Huckleberry': 29,
                                        'Kaki': 30,
                                        'Kiwi': 31,
                                        'Kumquats': 32,
                                        'Lemon': 33,
                                        'Lemon Meyer': 34,
                                        'Limes': 35,
                                        'Lychee': 36,
                                        'Mandarine': 37,
                                        'Mango': 38,
                                        'Mangostan': 39,
                                        'Melon Piel de Sapo': 40,
                                        'Mulberry': 41,
                                        'Nectarine': 42,
                                        'Orange': 43,
                                        'Papaya': 44,
                                        'Passion Fruit': 45,
                                        'Peach': 46,
                                        'Peach 2': 47,
                                        'Peach Flat': 48,
                                        'Pear': 49,
                                        'Pear Kaiser': 50,
                                        'Pineapple': 51,
                                        'Pineapple Mini': 52,
                                        'Pitahaya Red': 53,
                                        'Plum': 54,
                                        'Plum 2': 55,
                                        'Plum 3': 56,
                                        'Pomegranate': 57,
                                        'Pomelo Sweetie': 58,
                                        'Rambutan': 59,
                                        'Raspberry': 60,
                                        'Redcurrant': 61,
                                        'Strawberry': 62,
                                        'Strawberry Wedge': 63,
                                        'Tomato 1': 64,
                                        'Tomato 2': 65,
                                        'Tomato 4': 66,
                                        'Tomato Cherry Red': 67,
                                        'Tomato Maroon': 68,
                                        'Walnut': 69}
            self.label = list(self.Label_dict.keys())

        def read_images(self, filename_):
            self.cnt = len(os.listdir(self.img_dir))
            self.batch_holder = np.zeros((self.cnt, 100, 100, 3))
            for i, img in enumerate(os.listdir(self.img_dir)):
                img = image.load_img(os.path.join(self.img_dir, filename_), target_size=(100, 100))
                self.batch_holder[i, :] = img
            return self.batch_holder

        def predict(self):
            fig = plt.figure(figsize=(20, 20))
            for i, img in enumerate(self.batch_holder):
                fig.add_subplot(1, 1, i + 1)
                result = self.model.predict(self.batch_holder)
                result_classes = result.argmax(axis=-1)
                plt.title(self.label[result_classes[i]])
                plt.tick_params(
                    axis='both',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    labelleft=False)
                plt.imshow(img / 256.)
                break
            plt.show()

    print('display_image filename: ' + filename)
    print("directory is: ", os.getcwd())
    obj = Fruit(os.getcwd() + '\\static\\uploads\\')
    obj.read_images(filename)
    return obj.predict()


if __name__ == "__main__":
    app.run()
