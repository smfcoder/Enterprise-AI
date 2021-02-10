'''
Created on Dec 28, 2016

@author: hassan
'''
from nltk import *
from nltk.tokenize import sent_tokenize, word_tokenize

if __name__ == '__main__':
    
    with open('joyToken.txt', 'r') as myfile:
        joy = myfile.read()
    joySentences = sent_tokenize(joy)
    
    with open('saddnessToken.txt', 'r') as myfile:
        saddness = myfile.read()
    saddnessSentences = sent_tokenize(saddness)
    
    with open('shameToken.txt', 'r') as myfile:
        shame = myfile.read()
    shameSentences = sent_tokenize(shame)
    
    with open('lexicon_dictionary.txt', 'r') as myfile:
        lexicon_dictionary = myfile.read()
    lexicon_dictionary = lexicon_dictionary.split('\n')
    a = 0
    for x in lexicon_dictionary :
        lexicon_dictionary[a] = x.split(' ')
        a = a + 1
    
    s = LancasterStemmer()
    unwantedWordes = ['the' , 'a', 'is' , 'was' , 'are',
                      'were' , 'to', 'at', 'i' , 'my',
                      'on' , 'me'  , 'of' , '.' , 'in' ,
                      'that' , 'he' , 'she' , 'it' , 'by']
    
    if not os.path.isfile('featureVectors.csv'):
        open('featureVectors.csv' , 'w')   
    with open('featureVectors.csv', 'w') as featuresFile:
            featuresFile.write('')
            
    for i in range(0, a - 1): 
        lexicon_dictionary[i][0] = s.stem(lexicon_dictionary[i][0])
    
    # create feature Vector to each sentences in joy file
    for x in joySentences:
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
        featureVector.append(0)
        
        # write this feature vector to featureVectors File
        for k in range (0, 10):
            with open('featureVectors.csv', 'a') as featuresFile:
                featuresFile.write(str(featureVector[k]) + ',')
        with open('featureVectors.csv', 'a') as featuresFile:
            featuresFile.write(str(featureVector[10]) + '\n')
    
    # create feature Vector to each sentences in sadness file
    for x in saddnessSentences:
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
        featureVector.append(1)
        
        # write this feature vector to featureVectors File
        for k in range (0, 10):
            with open('featureVectors.csv', 'a') as featuresFile:
                featuresFile.write(str(featureVector[k]) + ',')
        with open('featureVectors.csv', 'a') as featuresFile:
            featuresFile.write(str(featureVector[10]) + '\n')
    
    
    # create feature Vector to each sentences in shame file
    
    for x in shameSentences:
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
        featureVector.append(2)
        
        # write this feature vector to featureVectors File
        for k in range (0, 10):
            with open('featureVectors.csv', 'a') as featuresFile:
                featuresFile.write(str(featureVector[k]) + ',')
        with open('featureVectors.csv', 'a') as featuresFile:
            featuresFile.write(str(featureVector[10]) + '\n')
    
    with open('featureVectors.csv', 'r') as myfile:
        features = myfile.read()
    features = features.split('\n')
    with open('featureVectors.csv', 'w') as featuresFile:
            featuresFile.write('')
    
    for i in range (0, 1040):
        with open('featureVectors.csv', 'a') as featuresFile:
            featuresFile.write(features[i] + '\n')
            featuresFile.write(features[i + 1040] + '\n')
            featuresFile.write(features[3117 - i] + '\n')
    
    
    pass
