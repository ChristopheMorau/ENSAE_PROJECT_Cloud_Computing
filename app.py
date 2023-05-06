"""Code for the API"""

from flask import Flask, render_template, request, send_file
import os, zipfile
import shutil


from main import generate_all

app = Flask(__name__)
INPUTPATH = './static/images_input/'

@app.route('/')
def upload():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def download():
    print('entering backend with following requested changes:', request.form.getlist('my_checkbox'))
    transformations_required = request.form.getlist('my_checkbox')
    if request.files:
        image = request.files["image"]
        input_images = [INPUTPATH + image for image in os.listdir(INPUTPATH)]
        if len(input_images) > 0:
            for input_images in input_images:
                if '.' in input_images[-7:-1]:
                    os.remove(input_images)   
                else:
                    shutil.rmtree(input_images, ignore_errors=True)
        
        if image.filename[-3:] == 'zip':
            print('unzipping dataset...')
            image.save('./static/images_input/image.zip')
            with zipfile.ZipFile('./static/images_input/image.zip', 'r') as zip_ref:
                zip_ref.extractall('./static/images_input/dataset') 
            os.remove('./static/images_input/image.zip')
            for img in os.listdir('./static/images_input/dataset'):
                shutil.copyfile('./static/images_input/dataset'+'/'+img,INPUTPATH+img)
            shutil.rmtree('./static/images_input/dataset', ignore_errors=True)
            
        else:
            image.save('./static/images_input/image.jpg')
        print('generating new dataset...')
        generate_all(transformations_required)
        name = './static/images_output'
        zip_name = name + '.zip'

        print('zipping new dataset...')
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            for folder_name, subfolders, filenames in os.walk(name):
                for filename in filenames:
                    file_path = os.path.join(folder_name, filename)
                    zip_ref.write(file_path, arcname=os.path.relpath(file_path, name))

        zip_ref.close()
        return render_template('results.html', user_image = "./static/images_output.zip")
    else:
        return 'nothing in files'

@app.route('/download',methods=["GET","POST"])
def downloadFile(): 
    image = request.form['foo']
    try:
       path = f'{image}'
       return send_file(path,mimetype='jpg', attachment_filename=image, as_attachment=True)
    except Exception as e:
        return str(e)
if __name__=='__main__':
    app.run()