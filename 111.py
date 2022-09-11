# coding=utf-8

import os
import importlib,sys
importlib.reload(sys)
import time
from flask import request, send_from_directory
from flask import Flask, request, redirect, url_for, render_template
import uuid
import tensorflow.compat.v1 as tf
FLAGS = tf.app.flags
from classify_image import run_inference_on_image
from classify_video import extract_video_keyframes
ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'jpeg', 'JPEG', 'png','mp4'])

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', '', """Path to graph_def pb, """)
tf.app.flags.DEFINE_string('model_name', 'my_inception_v4_freeze.pb', '')
tf.app.flags.DEFINE_string('label_file', 'label.txt', '')
tf.app.flags.DEFINE_string('upload_folder', './static', '')#path of pic
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

tf.app.flags.DEFINE_integer('port', '9865',
                            'server with port,if no port, use deault port 80')
tf.app.flags.DEFINE_boolean('debug', False, '')
UPLOAD_FOLDER = FLAGS.upload_folder

#文件格式限制
ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'jpeg', 'JPEG', 'png','mp4'])

app = Flask(__name__)
app._static_folder = UPLOAD_FOLDER

#该方法检验文件的后缀是否符合上面的要求
def allowed_files(filename):
    return '.' in filename and \
 \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def rename_filename(old_file_name):
    #找到最后一部分的文件名和后缀
    basename = os.path.basename(old_file_name)
    #文件名和后缀分离
    name, ext = os.path.splitext(basename)
    #uuid(1生成基于时间戳的随机数避免碰撞
    new_name = str(uuid.uuid1()) + ext

    return new_name


def inference(file_name):
    try:

        predictions, top_k, top_names = run_inference_on_image(file_name)

        print(predictions)

    except Exception as ex:

        print(ex)

        return ""

    new_url = './static/%s' % os.path.basename(file_name)

    image_tag = '<img src="%s"></img><p>'

    new_tag = image_tag % new_url

    format_string = ''

    for node_id, human_name in zip(top_k, top_names):
        score = predictions[node_id]

        format_string += '%s (score:%.5f)<BR>' % (human_name, score)

    ret_string = new_tag + format_string + '<BR>'

    return ret_string


@app.route("/", methods=['GET', 'POST'])
def root():
    result = render_template("xxx.html")

    if request.method == 'POST':

        file = request.files['file']
        print(file)

        old_file_name = file.filename
        print(old_file_name)

        if file and allowed_files(old_file_name):
            filename = rename_filename(old_file_name)
            print(filename)

            file_path = os.path.join(UPLOAD_FOLDER, filename)
            print(file_path)

            file.save(file_path)

            type_name = 'N/A'

            print('file saved to %s' % file_path)

            out_html = inference(file_path)

            return result + out_html

    return result


@app.route("/about", methods=['GET','POST'])
def about():
    return render_template("about.html")

@app.route("/answer", methods=['GET','POST'])
def answer():
    return render_template("answer.html")

@app.route("/index", methods=['GET','POST'])
def index():

    return render_template("index.html")
@app.route("/video", methods=['GET','POST'])
def video():
    result2 = render_template("video.html")
    if request.method == 'POST':

        file = request.files['file']
        print(file)

        old_file_name = file.filename
        print(old_file_name)
        print(os.path.abspath(old_file_name))
        old_file_name=os.path.abspath(old_file_name)
        print(old_file_name)
        old_file_name = old_file_name.replace('\\','/')
        print(old_file_name)
        file.save(old_file_name)
        extract_video_keyframes(old_file_name,r'./static')
        old_file_name='temporary-image-1.jpg'

        if file and allowed_files(old_file_name):
            filename = 'temporary-image-1.jpg'

            file_path = os.path.join(UPLOAD_FOLDER, filename)

            type_name = 'N/A'

            print('file saved to %s' % file_path)

            out_html = inference(file_path)

            return result2 + out_html

    return result2



if __name__ == "__main__":
    print('listening on port %d' % FLAGS.port)
    app.run(host='127.0.0.1', port=FLAGS.port, debug=FLAGS.debug, threaded=True)


