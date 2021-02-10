import os
from flask import Flask , redirect , render_template, request,url_for,session,flash,Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager , UserMixin , login_required ,login_user, logout_user,current_user
from werkzeug.utils import secure_filename
import time


UPLOAD_FOLDER = 'static/img/'
ALLOWED_EXTENSIONS = {'txt', 'png', 'jpg', 'jpeg', 'docx','wav'}


app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///db.db'
app.config['SECRET_KEY']='619619'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
db = SQLAlchemy(app)


login_manager = LoginManager()
login_manager.init_app(app)


class User(UserMixin,db.Model):
    id = db.Column(db.Integer,primary_key=True,autoincrement=True)
    username = db.Column(db.String(200), unique=True, nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password = db.Column(db.String(200))


class Imagee(db.Model):
    id = db.Column(db.Integer,primary_key=True,autoincrement=True)
    username = db.Column(db.String(200),nullable=False)
    email = db.Column(db.String(200),nullable=False)
    path = db.Column(db.String(200),nullable=False)
    place = db.Column(db.String(200), nullable=False)
    joy = db.Column(db.String(200),nullable=False)
    sad = db.Column(db.String(200), nullable=False)
    angry = db.Column(db.String(200), nullable=False)
    emotion = db.Column(db.String(200), nullable=False)
    review = db.Column(db.String(200), nullable=False)
    action_needed = db.Column(db.String(200), nullable=False)
    datafile = db.Column(db.String(), nullable=False)
    disgusting = db.Column(db.String(), nullable=False)
    neutral = db.Column(db.String(), nullable=False)
    fear = db.Column(db.String(), nullable=False)
    surprise = db.Column(db.String(), nullable=False)
    




@login_manager.user_loader
def get(id):
    return User.query.get(id)


@app.route('/home',methods=['GET'])
@login_required
def get_home():
    return render_template('index.html')


@app.route('/uploading/<id>',methods=['GET'])
@login_required
def get_uploading(id):
    return render_template('uploadimg.html',id=id)


@app.route('/upload',methods=['GET'])
@login_required
def upload_gets():
    return render_template('upload.html')


@app.route('/upload',methods=['POST'])
@login_required
def upload_post():
    pl = request.form['name']
    sad = 0
    joy = 0
    angry = 0
    fear=0
    disgusting=0
    neutral=0
    surprise=0
    emotion = ""
    datafile = ""
    review = ""
    action_needed = ""
    email = current_user.email
    username = current_user.username
    path = "static/img/"
    info = Imagee(username=username, email=email, path=path, place=pl, sad=sad, joy=joy, angry=angry,emotion=emotion,datafile=datafile,review=review,action_needed=action_needed,disgusting=disgusting,fear=fear,neutral=neutral,surprise=surprise)
    db.session.add(info)
    db.session.commit()
    id = info.id
    return redirect(url_for('get_uploading',id=id))


@app.route('/details/<id>',methods=['GET'])
@login_required
def get_details(id):
    information = Imagee.query.filter_by(id=id).first()
    
    

    total_count=int(information.sad)+int(information.angry)+int(information.joy)
    try:
        angry_per=round((int(information.angry)/total_count)*100,2)
    except:
        angry_per=0
    try:
        sad_per=round((int(information.sad)/total_count)*100,2)
    except:
        sad_per=0
    try:
        joy_per=round((int(information.joy)/total_count)*100,2)
    except:
        joy_per=0

    return render_template('details.html',**locals())


@app.route('/myuploads',methods=['GET'])
@login_required
def get_my_uploads():
    email = current_user.email
    table = Imagee.query.filter_by(email=email).all()
    return render_template('myuploads.html',table=table)


@app.route('/delete/<id>',methods=['GET'])
@login_required
def del_id(id):
    delete = Imagee.query.filter_by(id=id).first()
    db.session.delete(delete)
    db.session.commit()
    return redirect('/myuploads')



@app.route('/',methods=['GET'])
def get_login():
    return render_template('login.html')


@app.route('/signup',methods=['GET'])
def get_signup():
    return render_template('signup.html')


@app.route('/login',methods=['POST'])
def login_post():
    email = request.form['email']
    password = request.form['password']
    user = User.query.filter_by(email=email,password=password).first()
    if user:
        login_user(user)
        return redirect('/home')
    else:
        error ='Invalid'
        return render_template('login.html',error=error)


@app.route('/signup',methods=['POST'])
def signup_post():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    user = User(username=username,email=email,password=password)
    db.session.add(user)
    db.session.commit()
    user = User.query.filter_by(email=email).first()
    if user:
        login_user(user)
        return redirect('/home')
    else:
        error = 'Invalid'
        return render_template('signup.html', error=error)


# video analysis start
#from camera import VideoCamera

import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ds_factor=0.6

ang=0
dis=0
fear=0
happy=0
neut=0
sa=0
surp=0
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        ret,frame = self.video.read()
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        model.load_weights('video_analysis/model.h5')
        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)
            # dictionary which assigns each label an emotion (alphabetical order)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        global ang,dis,fear,happy,neut,sa,surp
        
        
        

        facecasc = cv2.CascadeClassifier('video_analysis/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if maxindex==0:
                ang+=1
            elif maxindex==1:
                dis+=1
            elif maxindex==2:
                fear+=1
            elif maxindex==3:
                happy+=1
            elif maxindex==4:
                neut+=1
            elif maxindex==5:
                sa+=1
            elif maxindex==6:
                surp+=1


        cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC)
        ret, frame = cv2.imencode('.jpg', frame)
        return frame.tobytes()




@app.route('/upload_video/<id>',methods=['GET'])
@login_required
def get_video_uploading(id):
    return render_template('video.html',id=id)
def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_upload',methods=['GET'])
@login_required
def upload_video_gets():
    return render_template('video_upload.html')


@app.route('/video_upload',methods=['POST'])
@login_required
def upload_video_post():
    pl = request.form['name']
    sad = 0
    joy = 0
    angry = 0
    disgusting=0
    fear=0
    surprise=0
    neutral=0
    emotion = ""
    review = ""
    action_needed = ""
    path=""
    datafile=""
    email = current_user.email
    username = current_user.username
    info = Imagee(username=username, email=email, place=pl, sad=sad, joy=joy, angry=angry,emotion=emotion,review=review,action_needed=action_needed,path=path,datafile=datafile,disgusting=disgusting,fear=fear,neutral=neutral,surprise=surprise)
    db.session.add(info)
    db.session.commit()
    id = info.id
    return redirect(url_for('get_video_uploading',id=id))

@app.route('/video_uploader', methods=['GET', 'POST'])
@login_required
def upload_video_file():
    
    if request.method == 'POST':
        

        if (happy+surp) > (sa+ang):
            review = "Interested"
            action_needed = "Give specific precautions and give best wishes for the use of product"
        elif neut > 0:
            review = "Interested"
            action_needed = "Ask the customer if he have any doubts about the product and solve them, give solutions for the customer's doubts."
        else:
            review = "Not interested"
            action_needed = "Ask customer for the problems he is facing, and give probable solution for the doubt."
        
        c_max=0
        emotion_detected=""
        if happy==sa==ang==dis==fear==neut==surp==0:
            emotion_detected="None"
            c_max=0
        elif happy > c_max:
            emotion_detected="Happy"
            c_max=happy
        elif sa > c_max:
            emotion_detected="Sad"
            c_max=sa
        elif ang > c_max:
            emotion_detected="Angry"
            c_max=ang
        elif dis > c_max:
            emotion_detected="Disgusting"
            c_max=dis
        elif fear > c_max:
            emotion_detected="Fear"
            c_max=fear
        elif neut > c_max:
            emotion_detected="Neutral"
            c_max=neut
        elif surp > c_max:
            emotion_detected="Surprise"
            c_max=surp

        id= request.form['id']
        data_row = Imagee.query.get(id)
        
        data_row.joy=happy
        data_row.sad=sa
        data_row.angry=ang
        data_row.disgusting=dis
        data_row.fear=fear
        data_row.neutral=neut
        data_row.surprise=surp

        data_row.review=review
        data_row.emotion=emotion_detected
        data_row.action_needed=action_needed
        db.session.commit()

        #global happy,sa,ang,dis,fear,neut,surp
        # happy=0
        # sa=0
        # ang=0
        # dis=0
        # fear=0
        # neut=0
        # surp=0
        #MESSAGE
        flash("Video Analysis Completed")
        return redirect(url_for('get_video_details', id=id))
    # else:
    #     delt = Imagee.query.get(id)
    #     db.session.delete(delt)
    #     db.session.commit()
    #     flash("Video Analysis not done")
    #     return redirect(url_for('upload_gets'))       

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/video_details/<id>',methods=['GET'])
@login_required
def get_video_details(id):
    information = Imagee.query.filter_by(id=id).first()
    
    

    total_count=int(information.sad)+int(information.angry)+int(information.joy)+int(information.disgusting)+int(information.fear)+int(information.surprise)+int(information.neutral)
    try:
        angry_per=round((int(information.angry)/total_count)*100,2)
    except:
        angry_per=0
    try:
        sad_per=round((int(information.sad)/total_count)*100,2)
    except:
        sad_per=0
    try:
        joy_per=round((int(information.joy)/total_count)*100,2)
    except:
        joy_per=0
    try:
        disgust_per=round((int(information.disgusting)/total_count)*100,2)
    except:
        disgust_per=0
    try:
        fear_per=round((int(information.fear)/total_count)*100,2)
    except:
        fear_per=0
    try:
        surprise_per=round((int(information.surprise)/total_count)*100,2)
    except:
        surprise_per=0
    try:
        neutral_per=round((int(information.neutral)/total_count)*100,2)
    except:
        neutral_per=0
    return render_template('video_details.html',**locals())


@app.route('/video_myuploads',methods=['GET'])
@login_required
def get_my_video_uploads():
    email = current_user.email
    table = Imagee.query.filter_by(email=email).all()
    return render_template('video_myuploads.html',table=table)


@app.route('/video_delete/<id>',methods=['GET'])
@login_required
def del_video_id(id):
    delete = Imagee.query.filter_by(id=id).first()
    db.session.delete(delete)
    db.session.commit()
    return redirect('/video_myuploads')

@app.route('/back/<id>',methods=['GET'])
@login_required
def del_video_analysis(id):
    delete = Imagee.query.filter_by(id=id).first()
    db.session.delete(delete)
    db.session.commit()
    return redirect('/home')




# video analysis end
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploader', methods=['GET', 'POST'])
@login_required

def upload_file():
    
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            name,ext=filename.split('.')
            
            if ext=='wav':
                import speech_recognition as sr
                from pydub import AudioSegment
                from fastpunct import FastPunct
                fastpunct = FastPunct('en')

                rec = sr.Recognizer()
                #filename = "static/audio/Record.wav"

                # name,ext = file.split('.')
                # sound=AudioSegment.from_mp3(file)
                # sound.export(name + '.wav',format='wav')

                
                with sr.AudioFile(path) as source:
                    audio = rec.record(source)  # read the entire audio file                  
                    transcript = rec.recognize_google(audio)

                print("Transcription: " + transcript)
                
                filename=name+".txt"
                fp=open("static/img/"+filename,"a")
                path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
                
                
                if len(transcript)>400:
                    split_strings = []
                    n  = 300
                    for index in range(0, len(transcript), n):
                        split_strings.append(transcript[index : index + n])    

                            
                    for sen in split_strings:
                        pun=fastpunct.punct([sen], batch_size=32)

                        print(pun)
                        chat=""
                        for i in pun:
                            chat+=i
                        fp.write(chat)
                else:
                    pun=fastpunct.punct([transcript], batch_size=32)
                    print(transcript)
                    chat=""
                    for i in pun:
                        chat+=i
                    fp.write(chat)
                fp.close()
                
                
            # audio file conversion to text end
            
            id= request.form['id']
            data_row = Imagee.query.get(id)
            data_row.path='static/img/'+filename
            db.session.commit()
            # audio file cnversion to text start
            
            
            #File Processing


            if os.stat(path).st_size > 0:

                from keras.models import model_from_json
                from tensorflow.python.keras.layers import Input, Dense
                from tensorflow import keras
                import numpy
                from nltk.tokenize import sent_tokenize, word_tokenize
                from nltk.stem.lancaster import LancasterStemmer


                # enter Your Sentences in Sentences.txt File
                # Only code needed to  Load Code
                json_file = open("model.json", 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                model = model_from_json(loaded_model_json)
                # load weights into new model
                model.load_weights("model.h5")
                print("Loaded model from disk")
                ###########################
                with open('lexicon_dictionary.txt', 'r') as myfile:
                    lexicon_dictionary = myfile.read()
                lexicon_dictionary = lexicon_dictionary.split('\n')
                a = 0

                for x in lexicon_dictionary:
                    lexicon_dictionary[a] = x.split(' ')
                    a = a + 1

                with open(path, 'r') as myfile:
                    sen = myfile.read()
                sentences = sent_tokenize(sen)

                if not os.path.isfile('featureVectorForSentence.csv'):
                    open('featureVectorForSentence.csv', 'w')
                with open('featureVectorForSentence.csv', 'w') as featuresFile:
                    featuresFile.write('')

                s = LancasterStemmer()
                unwantedWordes = ['the', 'a', 'is', 'was', 'are',
                                  'were', 'to', 'at', 'i', 'my',
                                  'on', 'me', 'of', '.', 'in',
                                  'that', 'he', 'she', 'it', 'by']
                for i in range(0, a - 1):
                    lexicon_dictionary[i][0] = s.stem(lexicon_dictionary[i][0])

                for x in sentences:
                    featureVector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    words = word_tokenize(x)
                    for y in words:
                        y = s.stem(y)
                        y = y.lower()
                        if y in unwantedWordes != -1:
                            continue
                        for i in range(0, a - 1):
                            if y == lexicon_dictionary[i][0]:
                                for j in range(0, 10):
                                    featureVector[j] = featureVector[j] + int(lexicon_dictionary[i][j + 1])
                                break
                    # write this feature vector to featureVectors File
                    for k in range(0, 9):
                        with open('featureVectorForSentence.csv', 'a') as featuresFile:
                            featuresFile.write(str(featureVector[k]) + ',')
                    with open('featureVectorForSentence.csv', 'a') as featuresFile:
                        featuresFile.write(str(featureVector[9]) + '\n')
                # to avoid one Sentence Error
                for k in range(0, 9):
                    with open('featureVectorForSentence.csv', 'a') as featuresFile:
                        featuresFile.write(str(featureVector[k]) + ',')
                with open('featureVectorForSentence.csv', 'a') as featuresFile:
                    featuresFile.write(str(featureVector[9]) + '\n')

                dataset = numpy.loadtxt("featureVectorForSentence.csv", delimiter=",")
                X = dataset[:-1, :]

                predictions = model.predict(X)
                rounded = numpy.around(predictions, decimals=0)
                print(rounded)

                c = 1
                shame = 0
                joy = 0
                sadness = 0
                print("Emotions For Each Sentences in sentences.txt File")
                for x in rounded:
                    if x[0] == 1 and x[1] == 0 and x[2] == 0:
                        joy = joy + 1
                        print("Sentence Number " + str(c) + " is JOY")
                    elif x[0] == 0 and x[1] == 1 and x[2] == 0:
                        sadness = sadness + 1
                        print("Sentence Number " + str(c) + " is Sadness")
                    elif x[0] == 0 and x[1] == 0 and x[2] == 1:
                        shame = shame + 1
                        print("Sentence Number " + str(c) + " is Shame")
                    c = c + 1
                print("Joy :" + str(joy))
                print("sadness :" + str(sadness))
                print("shame :" + str(shame))

                pass


                emotion_list=[sadness,shame,joy]
                high_value=0
                for i in range(len(emotion_list)):
                    if int(emotion_list[i]) > int(high_value):
                        high_value=emotion_list[i]
                i=emotion_list.index(high_value)
                
                
                if emotion_list[0]==emotion_list[1]==emotion_list[2]==str(0):
                    emotion_detected="No emotion"
                    review="Unknown"
                    action_needed="Please take review or feedback of customer"
                elif emotion_list[0]==emotion_list[1]==emotion_list[2]:
                    emotion_detected="Neutral"
                    review="Interested"
                    action_needed="Describe in Brief features of the product"
                elif emotion_list[0]==emotion_list[1]:
                    emotion_detected="fear"
                    review="Interested"
                    action_needed="Describe Positive/Fascinating features of the product"
                elif emotion_list[1]==emotion_list[2]:
                    emotion_detected="Puzzling"
                    review="Interested"
                    action_needed="Ask the queries, why the customer is getting puzzled about the products or refer similar products to the customer"
                elif emotion_list[0]==emotion_list[2]:
                    emotion_detected="Confused"
                    review="Interested"
                    action_needed="Get the doubts in the customer mind, refer customer similar or optional products for the selected product, Describe the features of product briefly"
                elif i==0:
                    emotion_detected="Sad"
                    review="Not interested"
                    action_needed="Create possible understandings about the product"
                elif i==1:
                    emotion_detected="Angry"
                    review="Not interested"
                    action_needed="Understand problem with the product review and give a proper solution"
                elif i==2:
                    emotion_detected="Happy"
                    review="Interested"
                    action_needed="Best Wishes!!!.Enjoy the product"
                
                
                with open(path, 'r') as myfile:
                    sen = myfile.read()
                

                val = Imagee.query.get(id)
                val.joy = joy
                val.sad = sadness
                val.angry = shame
                val.emotion = emotion_detected
                val.review = review
                val.action_needed = action_needed
                val.datafile = sen
                db.session.commit()




                #MESSAGE
                flash("File Uploaded")
                return redirect(url_for('get_details', id=id))

            else:

                os.remove("static/img/"+filename)
                delt = Imagee.query.get(id)
                db.session.delete(delt)
                db.session.commit()
                flash("File is empty")
                return redirect(url_for('upload_gets'))
        else:
            delt = Imagee.query.get(id)
            db.session.delete(delt)
            db.session.commit()
            flash("FILE UPLOADING FAILED")
            return redirect(url_for('upload_gets'))
    else:
        delt = Imagee.query.get(id)
        db.session.delete(delt)
        db.session.commit()
        flash("IMAGE UPLOADING FAILED")
        return redirect(url_for('upload_gets'))

    
@app.route('/logout',methods=['GET'])
def logout():
    logout_user()
    return redirect('/')


if __name__=='__main__':
    app.run(debug=True)