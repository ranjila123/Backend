import flask
import werkzeug
import time
import csv


import os


from flask import jsonify


from flask import send_file
# from flask import send_file, current_app as app
# from flask import send_from_directory, current_app as app

# from segment0220 import segment
from ScannedSegmentation import segment

app = flask.Flask(__name__)


UPLOAD_FOLDER = 'D:/Ranjila Main/Backend-SimpleHTR/src/UserForms'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    files_ids = list(flask.request.files)
    print("\nNumber of Received Images : ", len(files_ids))
    image_num = 1
    for file_id in files_ids:
        print("\nSaving Image ", str(image_num), "/", len(files_ids))
        imagefile = flask.request.files[file_id]
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        print("Image Filename : " + imagefile.filename)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'],timestr+'_'+filename))
        image_num = image_num + 1
    print("\n")


    status = segment()
    if(status==0):
        return "Error:Please upload NCE Admission Form."
        # return "Image Uploaded Successfully. Waiting for the result...."
        # response = {'message':'Image Uploaded Successfully. Waiting for the result....'} 

    else:
        return "Image Uploaded Successfully. Waiting for the result...."
        # response = {'message':'The given form is not NCE Admisson Form. Please upload correct form.'}
   
    # return jsonify(response)
    # return "Image Uploaded Successfully. Waiting for the result...."

# @app.route('/show/static-pdf/sample.pdf')
# def show_static_pdf():
#     with open('D:\MajorProject-master\src\sample.pdf', 'rb') as static_file:
#         return send_file(static_file, attachment_filename='file.pdf')










@app.route('/download_csv')
def download_csv():
    # Replace 'path/to/your/csv/file.csv' with the actual path to your CSV file
    return send_file('D:\MajorProject-master\src\output.csv', as_attachment=True)


input_file_path = "D:/Ranjila Main/Backend-SimpleHTR/src/output.csv"

with open(input_file_path, newline='') as input_file:
    reader = csv.reader(input_file)
    header_row = next(reader)
    input_file.seek(0)  # reset file pointer to beginning of file
    with open(input_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header_row)





app.run(host="0.0.0.0", port=8000, debug=True)



