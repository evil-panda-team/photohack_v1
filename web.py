from flask import render_template, request, flash, redirect
from flask import Flask
import os
from wtforms import StringField,Form, TextField, validators
from werkzeug.utils import secure_filename

app = Flask(__name__ )

UPLOAD_FOLDER = './static/src'
ALLOWED_EXTENSIONS = set(['ico', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

result_path = "static/res/server/"
src_img_path = "static/src/"

def get_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

class ReusableForm(Form):
    msg = StringField('Message:', validators=[validators.required()])
    selfie = StringField('Message:', validators=[validators.required()])


@app.route("/", methods=['GET', 'POST'])
def hello2():
    form = ReusableForm(request.form)
    server_url = get_ip() + ":5000"

    included_extensions = ['jpg', 'jpeg', 'bmp', 'png']
    images = [src_img_path + fn for fn in os.listdir(src_img_path)
                  if any(fn.endswith(ext) for ext in included_extensions)]

    result_images = []
    if request.method == 'POST':
        # if form.msg:
        #     print("MSG:", form.msg.data)
        # if form.selfie:
        #     print("SELFIE", form.selfie.data)

        for f in os.listdir(result_path):
            os.remove(result_path+f)

        import common
        common.fcn(form.selfie.data, form.msg.data, result_path="static/res/server/")

        result_images = [result_path + fn for fn in os.listdir(result_path)]

    return render_template('index.html', server_url=server_url, form=form, images = images, result_images=result_images)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploader', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect('')

