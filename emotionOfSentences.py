'''
Created on Dec 29, 2016

@author: hassan
'''
from keras.models import model_from_json
import numpy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import os

if __name__ == '__main__':  
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

    for x in lexicon_dictionary :
        lexicon_dictionary[a] = x.split(' ')
        a = a + 1
    
    with open('sentences.txt', 'r') as myfile:
        sen = myfile.read()
    sentences = sent_tokenize(sen)
    
    if not os.path.isfile('featureVectorForSentence.csv'):
        open('featureVectorForSentence.csv' , 'w')   
    with open('featureVectorForSentence.csv', 'w') as featuresFile:
            featuresFile.write('')
    
    s = LancasterStemmer()
    unwantedWordes = ['the' , 'a', 'is' , 'was' , 'are',
                      'were' , 'to', 'at', 'i' , 'my',
                      'on' , 'me'  , 'of' , '.' , 'in' ,
                      'that' , 'he' , 'she' , 'it' , 'by']
    for i in range(0, a - 1): 
        lexicon_dictionary[i][0] = s.stem(lexicon_dictionary[i][0])
    
    for x in sentences:
        featureVector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        words = word_tokenize(x)
        for y in words :
            y = s.stem(y)
            y = y.lower()
            if y in unwantedWordes != -1:
                continue
            for i in range(0, a - 1): 
                if y == lexicon_dictionary[i][0]:
                    for j in range(0, 10):
                        featureVector[j] = featureVector[j] + int (lexicon_dictionary[i][j + 1])
                    break
        # write this feature vector to featureVectors File
        for k in range (0, 9):
            with open('featureVectorForSentence.csv', 'a') as featuresFile:
                featuresFile.write(str(featureVector[k]) + ',')
        with open('featureVectorForSentence.csv', 'a') as featuresFile:
            featuresFile.write(str(featureVector[9]) + '\n')
    # to avoid one Sentence Error
    for k in range (0, 9):
        with open('featureVectorForSentence.csv', 'a') as featuresFile:
                featuresFile.write(str(featureVector[k]) + ',')
    with open('featureVectorForSentence.csv', 'a') as featuresFile:
        featuresFile.write(str(featureVector[9]) + '\n')
        
    
    dataset = numpy.loadtxt("featureVectorForSentence.csv", delimiter=",")
    X = dataset[ :-1, :]
    
    predictions = model.predict(X)
    rounded = numpy.around(predictions, decimals=0)
    print (rounded)
    
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
    print ("Joy :" + str(joy))
    print ("sadness :" + str(sadness))
    print ("shame :" + str(shame))
    
    
    pass
