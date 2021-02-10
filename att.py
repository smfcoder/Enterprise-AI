import speech_recognition as sr
from pydub import AudioSegment
from fastpunct import FastPunct
fastpunct = FastPunct('en')

rec = sr.Recognizer()
filename = "static/audio/Record.wav"

# name,ext = file.split('.')
# sound=AudioSegment.from_mp3(file)
# sound.export(name + '.wav',format='wav')


with sr.AudioFile(filename) as source:
    audio = rec.record(source)  # read the entire audio file                  
    transcript = rec.recognize_google(audio)

print("Transcription: " + transcript)
ftrans=""
if len(transcript)>400:
    split_strings = []
    n  = 300
    for index in range(0, len(transcript), n):
        split_strings.append(transcript[index : index + n])    

fp=open("static/audio/record.txt","a")
for sen in split_strings:
    pun=fastpunct.punct([sen], batch_size=32)

    print(pun)
    chat=""
    for i in pun:
        chat+=i
    fp.write(chat)
fp.close()