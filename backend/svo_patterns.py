from subject_verb_object_extract import findSVOs, printDeps, nlp
import nltk 
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from nltk import pos_tag
def decomposotionality(doc): # return decompsitional elements for p
    sub=[]
    obj=[]
    opinion=[]

    for token in doc:

        if (token.dep_=='nsubj'):
            sub.append(token.text)
        # extract object
        elif (token.dep_=='pobj' or token.dep_=='dobj'):
            #print(token.text)
            obj.append(token.text)
        elif (token.dep_=='amod' or token.dep_=='acomp' or token.dep_=='ROOT'):

            opinion.append(token.text)
    return sub,opinion,obj

def get_componenets_svo(input_text):
    target_concept,aspect,opinion=[],[],[]

    doc1 = nlp(input_text)
    #doc2 = nlp(text2)    
    more_elements1=decomposotionality(doc1)
    #more_elements2=decomposotionality(doc2)
    tok1 = nlp(input_text)
    svos1 = findSVOs(tok1)  
    comp1=[list(elem) for elem in svos1]    

    more_elements1=[list(elem) for elem in more_elements1]  
    #print(f'component svo: {comp1},more elements: {more_elements1}')
    
    for cmp1 in comp1:
        print(len(cmp1))
        if len(cmp1)==3:
            target_concept.append(cmp1[0])
            opinion.append(cmp1[1])
            aspect.append(cmp1[2])
        if len(cmp1)==2:
            target_concept.append(cmp1[0])
            opinion.append(cmp1[1])
            #aspect.append(cmp1[2])
    #print("the  three elements : ", "tc: ",target_concept,"asp: ",aspect,"opn: ",opinion)        
    comp1=comp1+more_elements1####elements of p1
    comp11=[]  ## remve stop words from p2  
    for ll in comp1:# list
        wrd_lst=[]

        for phrase in ll: #pharse
            phrase_wrd=""

            tokens=phrase.split()
            for word in tokens:
                if word not in stopwords.words('english'):
                    if word:
                        #print("wordsssssss",word)
                        phrase_wrd=phrase_wrd+" "+ word.strip()
            if phrase_wrd:
                wrd_lst.append(phrase_wrd.strip())
        comp11.append(wrd_lst)

            
    return target_concept,aspect,opinion