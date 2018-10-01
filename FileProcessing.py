import glob
import os
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import collections
import math
from collections import OrderedDict

##  Name - Abhishek Ranjan
##  UIN - 657657618
##  NetId - aranja8

path = 'cranfieldDocs/*'

files = glob.glob(path)

queryList = []
queryListEnh = []
stops = []
queryList = []
queryListNew = []
fileList = []
indexList = {}
wordslistVocab = collections.defaultdict(list)
wordslistDoc = []
vocabidf = {}
sortedqryDoc = {}

#########Porter Stemmer #####################
def word_stemming(word):
     return ps.stem(word)
 
######### Function for Punctuation Removal #####################
def punctuation_remove(word):
    punctuations = '''!()-[]{};:'"\,<>./?@#+=$%^&*_~'''
    # remove punctuation from the string
    no_punct = ""
    for char in word:
        if char not in punctuations and not char.isdigit():
            no_punct = no_punct + char
    return no_punct
 
########## Function for tokenizing Query Document###################
def wordtokenizeQ(queryList, queryListEnh):
    for item in queryList:
        itemListNew = []
        for word in item:
            ## Call to punctuation removal ##
            word = punctuation_remove(word)
            if word != '' and word.lower() not in stops:
                wordNew  = word_stemming(word)                
                itemListNew.append(wordNew.lower())
        queryListEnh.append(itemListNew)
    

########## Function to create Inverted Document Frequency ###################
def wordtokenizeD(wordslistInp):
    i = 0
    for item in wordslistInp:
        itemDict = {}
        count = 1
        for word in item:
            ## Call to punctuation removal ##
            word = punctuation_remove(word)            
            if word != '' and word.lower() not in stops: 
                wordNew  = word_stemming(word)
                if wordNew.lower() in itemDict:
                    itemDict[wordNew.lower()] += 1.0
                    if(itemDict[wordNew.lower()] > count):
                        count = itemDict[wordNew.lower()]
                else:
                    itemDict[wordNew.lower()] = 1.0
                    #count += 1
        for term in itemDict:
            itemapnd = {}
            #itemapnd[i] = itemDict[term]/len(itemDict)
            #itemapnd[files[i]] = itemDict[term]/count
            itemapnd[i] = itemDict[term]/count
            if term in wordslistVocab:
                listTemp = []
                listTemp = wordslistVocab[term]
                listTemp.append(itemapnd)
                wordslistVocab[term] = listTemp
            else:
                listTemp = []
                listTemp.append(itemapnd)
                wordslistVocab[term] = listTemp                
        i += 1
        

######### Function for Processing Documents ##################
for paths in files:
    with open(os.path.join(os.path.dirname(__file__), paths)) as file:
        fileList.append(paths)
        strInp = ""
        text = file.read()
        soup = BeautifulSoup(text, 'html.parser')
        span = soup.findAll(['title','text'])
        for th in span:
            strInp += th.text
        wordslistDoc.append(strInp.split())
        
######### Function for Processing on Query ##################
with open('queries.txt', 'r') as queryFile:
    queryListInp = queryFile.read().split('\n')
    for item in queryListInp:
        queryList.append(item.split())
        
########### Reading StopWords ###################################################
with open('stopwords.txt', 'r') as stopFile:
    stops = stopFile.read().split('\n')
        
ps = PorterStemmer() 

########### Pre-Processing on Query ###############################
wordtokenizeQ(queryList, queryListEnh)

########### Pre-Processing on Document ###############################
wordtokenizeD(wordslistDoc)


########## Updating Inverted Documents List, Storinf tf-idf value / Creating idf value for all words ########################################
for item in wordslistVocab:
    listTemp = []
    listTemp = wordslistVocab[item]
    matchdoc = len(listTemp)
    totaldoc = len(files)
    idf = math.log2(totaldoc/matchdoc)
    vocabidf[item] = idf
    for listItem in listTemp:
        for key, value in listItem.items():
            val = listItem[key]
            listItem[key] = idf*val
    wordslistVocab[item] = listTemp


########## Calculating Vector Length for all Document ########################################
SumDoc = {}
#count = {}
for item in wordslistVocab:
    listTemp = []
    listTemp = wordslistVocab[item]
    for listItem in listTemp:
        for key, value in listItem.items():
            if key in SumDoc:
                valOld = SumDoc[key]
                SumDoc[key] = valOld + (value*value)
                #count[key] += 1
            else:
                SumDoc[key] = (value*value)
                #count[key] = 1
for keyOld, valueOld in SumDoc.items():
    valUpdt = math.sqrt(valueOld)
    SumDoc[keyOld] = valUpdt
    
                

########## Updating Inverted Documents List, Storinf tf-idf value / Creating idf value for all words ########################################
for item in wordslistVocab:
    listTemp = []
    listTemp = wordslistVocab[item]
    matchdoc = len(listTemp)
    totaldoc = len(files)
    idf = math.log10(totaldoc/matchdoc)
    vocabidf[item] = idf
    for listItem in listTemp:
        for key, value in listItem.items():
            val = listItem[key]
            listItem[key] = idf*val


########## Calculating Relevant Document List ########################################            
relevDocOut = OrderedDict()
SumQry = {}
i = 0
for item in queryListEnh:    
    itemLen  =len(item)
    relevdoc = {}
    SumQry[i] = 0.0    
    for word in item:
        if word in wordslistVocab:
            tfword = 1.0
            idfword = vocabidf[word]
            prodtfidf = tfword*idfword
            valOld = SumQry[i]
            SumQry[i] = valOld + (prodtfidf*prodtfidf)
            doclist =  wordslistVocab[word]
            for listItem in doclist:
                 for key, value in listItem.items():
                     if key not in relevdoc:
                         val = listItem[key]
                         relevdoc[key] = (val * prodtfidf)
                     else:
                         valOld = relevdoc[key]
                         val = listItem[key]
                         relevdoc[key] = valOld + (val * prodtfidf)
    
    #relevDocOut[queryListInp[i]] = relevdoc
    relevDocOut[i] = relevdoc
    i += 1

################Updating the relevance doc value by dividing it with denominator ######################
print("\n")
for qrykey, qryVal in relevDocOut.items():
    print("For Query - "+ str(qrykey) +"\n")
    print("Number of matching Documents")
    print(len(qryVal))
    print("\n")
    for dockey, docVal in qryVal.items():
        valOld = qryVal[dockey]
        valUpdt = valOld/ ((SumDoc[dockey]) * (math.sqrt(SumQry[qrykey])))
        relevDocOut[qrykey][dockey] = valUpdt
        
################ Fetching Relevant Document #########################################################
relevDocInTemp = []
relevDocIn = {}
i = 1
with open('relevance.txt', 'r') as relevFile:
    relevListInp = relevFile.read().split('\n')
    for item in relevListInp:
        relevDocInTemp.append(item.split())
    listTemp = []
    for place in relevDocInTemp:
        if(place[0] == str(i)):
            fileno = int(place[1]) - 1
            listTemp.append(fileno)
        else:
            relevDocIn[i-1] = listTemp
            listTemp = []
            i += 1
            fileno = int(place[1]) - 1
            listTemp.append(fileno)
    relevDocIn[i-1] = listTemp

            
 
################ Calculating Precision Recall ###################################################
def precisonRecall(relevDocOut, relevDocIn, numDocs):
    
    n = len(relevDocOut)
    prcSum = 0.0
    recSum = 0.0
    print("\n")
    print("\n")
    print("**************************************")
    print("Stat for Number of Document " + str(numDocs))
    for qryId, docList in relevDocOut.items():
        sortedDocList = []
        #for dockey, docVal in docList.items():
        sortedDocList = sorted(docList.items(), key=lambda kv: kv[1], reverse = True)
        sortedqryDoc[qryId] = sortedDocList
    for i in range(len(relevDocOut)):
        listTemp = []
        finalList = []
        listMtch = []
        listTemp = sortedqryDoc[i][:numDocs]
        listMtch = [x[0] for x in listTemp]
        relevList =  relevDocIn[i]
        finalList = [x for x in listMtch if x in relevList]
        numRlvntDoc = len(relevDocIn[i])
        numMtchDoc = len(finalList)
        Recall = numMtchDoc/numRlvntDoc
        Precision = numMtchDoc/numDocs
        print("Precision for Query - "+ str(i) + " --> " + str(Precision))
        print("Recall for Query - "+ str(i) + "--> " + str(Recall))
        prcSum += Precision
        recSum += Recall
    print("\nAverage Precision for "+ str(numDocs) +" Documents- " + " --> " + str(prcSum/n))
    print("Average Recall for "+ str(numDocs) +" Documents- "+ "--> " + str(recSum/n))
        
        
       
            
        
    
    
precisonRecall(relevDocOut, relevDocIn, 10)
precisonRecall(relevDocOut, relevDocIn, 50)
precisonRecall(relevDocOut, relevDocIn, 100)
precisonRecall(relevDocOut, relevDocIn, 500)
            
    

            

    
    
    






        
            

        
        
        
        
    

        
    
    
    



