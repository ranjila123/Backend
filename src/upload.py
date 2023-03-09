import os
from datetime import datetime
from flask import Flask, request, render_template, send_file
from main import infer_by_web
from ScannedSegmentation import segment
# from segment0220 import segment
import csv

__author__ = 'Romu'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__)) # project abs path



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_page", methods=["GET"])
def upload_page():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    # folder_name = request.form['uploads']
    target = os.path.join(APP_ROOT, 'UserForms/')
    # print('target')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    print(request.files.getlist("file"))
    option = request.form.get('optionsPrediction')
    print("Selected Option:: {}".format(option))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if (ext == ".jpg") or (ext == ".png"):
            print("File supported moving on...")
        else:
            render_template("Error.html", message="Files uploaded are not supported...")
        savefname = datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + "."+ext
        destination = "/".join([target, savefname])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)
        segment()
        result = True
    return render_template("upload.html")

    
# @app.route("/finish")
# def finish():

#     return render_template("complete.html", image_name=savefname, result=result)

# @app.route('/finish')
# def finish():
#     return render_template("complete.html")




@app.route('/finish')
def download_file():
	path = "D:/Ranjila Main/Backend-SimpleHTR/src/output.csv"
	#path = "sample.txt"
	return send_file(path, as_attachment=True)


input_file_path = "D:/Ranjila Main/Backend-SimpleHTR/src/output.csv"

with open(input_file_path, newline='') as input_file:
    reader = csv.reader(input_file)
    header_row = next(reader)
    input_file.seek(0)  # reset file pointer to beginning of file
    with open(input_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header_row)





if __name__ == "__main__":
    app.run(port=5000, debug=True)