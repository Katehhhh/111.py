# -*- coding: utf-8 -*-
# coding=utf-8

import os
import importlib,sys
import subprocess
importlib.reload(sys)
from flask import Flask, request, render_template
import uuid
import tensorflow.compat.v1 as tf
import pandas as pd
import time
FLAGS = tf.app.flags
from classify_image import run_inference_on_image
from classify_video import extract_video_keyframes

#文件格式限制
ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'jpeg', 'JPEG', 'png','mp4','wav','mp3'])

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', '', """Path to graph_def pb, """)
tf.app.flags.DEFINE_string('model_name', 'my_inception_v4_freeze.pb', '')
tf.app.flags.DEFINE_string('label_file', 'label.txt', '')
tf.app.flags.DEFINE_string('upload_folder', './static', '')#path of pic
tf.app.flags.DEFINE_string('audio_upload_folder', './example', '')#path of audio
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

tf.app.flags.DEFINE_integer('port', '9865',
                            'server with port,if no port, use deault port 80')
tf.app.flags.DEFINE_boolean('debug', True, '')
UPLOAD_FOLDER = FLAGS.upload_folder
Audio_UPLOAD_FOLDER = FLAGS.audio_upload_folder

app = Flask(__name__)
app._static_folder = UPLOAD_FOLDER


def run(cmd, shell=False) -> (int, str):
    """
    开启子进程，执行对应指令，控制台打印执行过程，然后返回子进程执行的状态码和执行返回的数据
    :param cmd: 子进程命令
    :param shell: 是否开启shell
    :return: 子进程状态码和执行结果
    """
    print('\033[1;32m************** START **************\033[0m') # 使用绿色字体
    p = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result = []
    while p.poll() is None:
        line = p.stdout.readline().strip()
        if line:
            line = _decode_data(line)
            result.append(line)
            print('\033[1;35m{0}\033[0m'.format(line))
        # 清空缓存
        sys.stdout.flush()
        sys.stderr.flush()
    # 判断返回码状态
    if p.returncode == 0:
        print('\033[1;32m************** SUCCESS **************\033[0m')
    else:
        print('\033[1;31m************** FAILED **************\033[0m')
    return p.returncode, '\r\n'.join(result)


def _decode_data(byte_data: bytes):
    try:
        return byte_data.decode('UTF-8')
    except UnicodeDecodeError:
        return byte_data.decode('GB18030')


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

@app.route("/test", methods=['GET','POST'])
def test():
    result = render_template("test.html")
    if request.method == 'POST':
        file = request.files['file']
        file_name = file.filename
        args = request.form
        print(args)
        cmd = 'python analyze.py --locale zh '
        for arg in args:
            cmd += '--'+ arg +' '+ args[arg] +' '
        print(cmd)
        print(file_name)
        base_name = os.path.splitext(file_name)[0]
        print(base_name)
        if file and allowed_files(file_name):
            filename = file_name
            file_path = os.path.join(Audio_UPLOAD_FOLDER, filename)
            print(Audio_UPLOAD_FOLDER)
            print(file_path)
            file.save(file_path)
            type_name = 'N/A'
            print('file saved to %s' % file_path)
            # 在这里启动新的进程，阻塞，仅仅给输出提供异步方法
            return_code, data = run(cmd)
            print('return code:', return_code,'data:', data)
            csv_name = Audio_UPLOAD_FOLDER+'/'+base_name+'.BirdNET.results.csv'
            csv_base_name = base_name+'.BirdNET.results.csv'
            # 读取csv文件
            while not os.path.isfile(Audio_UPLOAD_FOLDER+'/'+base_name+'.BirdNET.results.csv'):
                time.sleep(100)
                print("Please Wait")
            data = pd.read_csv(Audio_UPLOAD_FOLDER+'/'+base_name+'.BirdNET.results.csv', sep=',',encoding='gbk')
            start = data["Start (s)"]
            end = data["End (s)"]
            CN = data["Common name"]
            conf = data["Confidence"]
            format_string = ''
            for i in range(len(start)):
                format_string += f'{start[i]} , {end[i]}, {CN[i]}, {conf[i]}<BR>'
            ret_string = format_string + '<BR>'
            os.remove(Audio_UPLOAD_FOLDER+'/'+file_name)
            os.remove(csv_name)
            return result+ret_string
    return result

if __name__ == "__main__":
    print('listening on port %d' % FLAGS.port)
    app.run(host='127.0.0.2', port=FLAGS.port, debug=FLAGS.debug, threaded=True)


