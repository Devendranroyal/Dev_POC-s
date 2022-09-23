from flask import Flask, render_template, request, flash
import os
from recognize import alpr_function
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'temp'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'some sort of thing'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def mainApp():
    pltInfo = None
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            pltInfo = alpr_function(os.path.join(app.config['UPLOAD_FOLDER'], filename)) # ['123', 'KKK']
            pltInfo.sort(key=lambda pltLen: len(pltLen), reverse=True)
            print(pltInfo)
            if len(pltInfo) > 0:
                flash(pltInfo[0])
            else:
                flash("No Readings !")
    return render_template("ALPR.html", pltInfo=pltInfo)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
##Flask script contain an environment variable