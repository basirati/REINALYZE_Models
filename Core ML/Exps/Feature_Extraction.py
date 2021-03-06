#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.corpus import wordnet


# In[2]:


model='en_core_web_sm'
nlp = spacy.load(model)
nltk.download('stopwords')


# In[3]:


def getOrigDep(word):
    if (word.dep_ != 'conj'):
        return word.dep_
    return getOrigDep(word.head)


# In[4]:


def getSynAnt(word):
    synonyms = [] 
    antonyms = []
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonyms.append(l.name()) 
            if l.antonyms(): 
                antonyms.append(l.antonyms()[0].name()) 
    return synonyms, antonyms


# In[5]:


def getWUPSimilarity(w1, w2):
    doc1 = nlp(w1)
    doc2 = nlp(w2)
    if doc1[0].lemma_ == doc2[0].lemma_:
        return 1
    synonyms, _ = getSynAnt(w1)
    if w2 in synonyms:
        return 0.9
    synonyms, _ = getSynAnt(w2)
    if w1 in synonyms:
        return 0.9

    #NOUN
    synw1s = wordnet.synsets(w1, wordnet.NOUN)
    if len(synw1s) > 0:
        synw2s = wordnet.synsets(w2, wordnet.NOUN)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])
        synw2s = wordnet.synsets(w2, wordnet.VERB)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])
        synw2s = wordnet.synsets(w2, wordnet.ADJ)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])
        synw2s = wordnet.synsets(w2, wordnet.ADV)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])
    #VERB
    synw1s = wordnet.synsets(w1, wordnet.VERB)
    if len(synw1s) > 0:
        synw2s = wordnet.synsets(w2, wordnet.NOUN)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])
        synw2s = wordnet.synsets(w2, wordnet.VERB)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])
        synw2s = wordnet.synsets(w2, wordnet.ADJ)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])
        synw2s = wordnet.synsets(w2, wordnet.ADV)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])
    #ADJ
    synw1s = wordnet.synsets(w1, wordnet.ADJ)
    if len(synw1s) > 0:
        synw2s = wordnet.synsets(w2, wordnet.NOUN)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])
        synw2s = wordnet.synsets(w2, wordnet.VERB)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])
        synw2s = wordnet.synsets(w2, wordnet.ADJ)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])
        synw2s = wordnet.synsets(w2, wordnet.ADV)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])
    #ADV
    synw1s = wordnet.synsets(w1, wordnet.ADV)
    if len(synw1s) > 0:
        synw2s = wordnet.synsets(w2, wordnet.NOUN)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])
        synw2s = wordnet.synsets(w2, wordnet.VERB)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])
        synw2s = wordnet.synsets(w2, wordnet.ADJ)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])
        synw2s = wordnet.synsets(w2, wordnet.ADV)
        if len(synw2s) > 0:
            return synw1s[0].wup_similarity(synw2s[0])


# In[6]:


def getAntonymity(w1, w2): 
    antonyms = [] 
    for syn in wordnet.synsets(w1): 
        for l in syn.lemmas(): 
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name()) 
    if len(antonyms) < 1:
        return 0
    if w2 in antonyms:
        return 1
    avg = 0
    for a in antonyms:
        tmp = getWUPSimilarity(w2, a)
        if tmp == None:
            tmp = 0
        avg = avg + tmp
    if len(antonyms) > 0:
        avg = avg / len(antonyms)
    else:
        return 0
    return avg


# In[7]:


def normalizeSimAnt(sim):
    if sim == None:
        sim = 'nothing'
    elif sim < 0.2:
        sim = 'low'
    elif 0.2 <= sim < 0.4:
        sim = 'some'
    elif 0.4 <= sim < 0.6:
        sim = 'medium'
    elif 0.6 <= sim < 0.8:
        sim = 'much'
    elif 0.8 <= sim < 0.95:
        sim = 'high'
    elif 0.95 <= sim:
        sim = 'strong'
        
    return sim
    


# In[8]:


def getStcSubObjFeatures(s1, s2):
    doc1 = nlp(s1)
    doc2 = nlp(s2)
    sfts = {'subjsubj' : 0, 'subjsubjn': 0, 'subjobj' : 0, 'subjobjn' : 0, 'objsubj' : 0, 'objsubjn' : 0, 'objobj' : 0, 'objobjn' : 0}
    afts = {'asubjsubj' : 0, 'asubjsubjn': 0, 'asubjobj' : 0, 'asubjobjn' : 0, 'aobjsubj' : 0, 'aobjsubjn' : 0, 'aobjobj' : 0, 'aobjobjn' : 0}
    state = 'none'
    for t1 in doc1:
        if getOrigDep(t1).endswith('subj'):
            state = 'subj'
        elif getOrigDep(t1).endswith('obj'):
            state = 'obj'
        else:
            continue
        state2 = 'none'
        for t2 in doc2:
            if getOrigDep(t2).endswith('subj'):
                state2 = 'subj'
            elif getOrigDep(t2).endswith('obj'):
                state2 = 'obj'
            else:
                continue
            #similarity = normalizeSimAnt(getWUPSimilarity(t1.text, t2.text))
            #antonymity = normalizeSimAnt(getAntonymity(t1.text, t2.text))
            #if similarity != 'nothing' and similarity != 'low':
            #    features.append('SIMILAR_' + state + state2 + '_' + similarity)
            #if antonymity != 'nothing' and antonymity != 'low':
            #    features.append('ANTONYM_' + state + state2 + '_' + antonymity)
            similarity = getWUPSimilarity(t1.text, t2.text)
            if similarity == None:
                similarity = 0
            antonymity = getAntonymity(t1.text, t2.text)
            sfts[state+state2] = sfts[state+state2] + similarity
            sfts[state+state2+'n'] = sfts[state+state2+'n'] + 1
            afts['a'+state+state2] = afts['a'+state+state2] + antonymity
            afts['a'+state+state2+'n'] = afts['a'+state+state2+'n'] + 1
    
    if sfts['subjsubj' + 'n'] > 0:
        sfts['subjsubj'] = normalizeSimAnt(sfts['subjsubj'] / sfts['subjsubj' + 'n'])
    if sfts['subjobj' + 'n'] > 0:
        sfts['subjobj'] = normalizeSimAnt(sfts['subjobj'] / sfts['subjobj' + 'n'])
    if sfts['objsubj' + 'n']  > 0:
        sfts['objsubj'] = normalizeSimAnt(sfts['objsubj'] / sfts['objsubj' + 'n'])
    if sfts['objobj' + 'n'] > 0:
        sfts['objobj'] = normalizeSimAnt(sfts['objobj'] / sfts['objobj' + 'n'])
    
    if afts['asubjsubj' + 'n'] > 0:
        afts['asubjsubj'] = normalizeSimAnt(afts['asubjsubj'] / afts['asubjsubj' + 'n'])
    if afts['asubjobj' + 'n'] > 0:
        afts['asubjobj'] = normalizeSimAnt(afts['asubjobj'] / afts['asubjobj' + 'n'])
    if afts['aobjsubj' + 'n'] > 0:
        afts['aobjsubj'] = normalizeSimAnt(afts['aobjsubj'] / afts['aobjsubj' + 'n'])
    if afts['aobjobj' + 'n'] > 0:
        afts['aobjobj'] = normalizeSimAnt(afts['aobjobj'] / afts['aobjobj' + 'n'])
    
    sfts.update(afts)
    return sfts
    


# In[9]:


def normalizeSimAnt2(sim):
    if sim == None:
        sim = 0
    elif sim < 0.2:
        sim = 0
    elif 0.2 <= sim < 0.8:
        sim = 0.5
    elif 0.8 <= sim:
        sim = 1
        
    return sim


# In[10]:


def getStcSubObjFeatures2(s1, s2):
    doc1 = nlp(s1)
    doc2 = nlp(s2)
    sfts = {'subjsubj' : 0, 'subjobj' : 0, 'objsubj' : 0, 'objobj' : 0}
    afts = {'asubjsubj' : 0, 'asubjobj' : 0, 'aobjsubj' : 0, 'aobjobj' : 0}
    state = 'none'
    for t1 in doc1:
        if getOrigDep(t1).endswith('subj'):
            state = 'subj'
        elif getOrigDep(t1).endswith('obj'):
            state = 'obj'
        else:
            continue
        state2 = 'none'
        for t2 in doc2:
            if getOrigDep(t2).endswith('subj'):
                state2 = 'subj'
            elif getOrigDep(t2).endswith('obj'):
                state2 = 'obj'
            else:
                continue
          
            similarity = normalizeSimAnt2(getWUPSimilarity(t1.text, t2.text))
            antonymity = normalizeSimAnt2(getAntonymity(t1.text, t2.text))
            if similarity > 0:
                if sfts[state+state2] < similarity:
                    sfts[state+state2] = similarity
            if antonymity > 0:
                    if afts['a'+state+state2] < antonymity:
                        afts['a'+state+state2] = antonymity
    
    sfts.update(afts)
    return sfts


# In[11]:



def isStopWord(w):
    if w.pos_ == 'DET':
        return True
    if w in set(nltk.corpus.stopwords.words('english')):
        return True
    return False


# In[12]:


def getNounLemmas(s):
    res = []
    doc = nlp(s)
    for nc in doc.noun_chunks:
        for t in nc:
            if not(isStopWord(t)):
                res.append(t.lemma_)
    return res


# In[13]:


def getNounSimilarityPortion(r1, r2):
    ns1 = getNounLemmas(r1)
    ns2 = getNounLemmas(r2)
    if len(ns1) < 1 or len(ns2) < 1:
        return 'low'
    num1 = 0
    for n in ns1:
        for m in ns2:
            if n == m:
                num1 = num1 + 1
                break
    if len(ns1) > 0:
        ovlap = num1/len(ns1)
    else:
        ovlap = 0
    if ovlap < 0.2:
        ovlap = 0
    elif 0.2 <= ovlap < 0.4:
        ovlap = 0.3
    elif 0.4 <= ovlap < 0.6:
        ovlap = 0.5
    elif 0.6 <= ovlap < 0.8:
        ovlap = 0.7
    elif 0.8 <= ovlap:
        ovlap = 1
    return ovlap


# In[14]:


def getNSPFeatures(r1, r2):
    res = {'noun_ovlap_1_2' : getNounSimilarityPortion(r1, r2), 'noun_ovlap_2_1' : getNounSimilarityPortion(r2, r1)}
    return res


# In[15]:


def getVerbLemmas(s):
    res = []
    doc = nlp(s)
    for t in doc:
        if t.tag_.startswith('VB'):
            res.append(t.lemma_)
    return res


# In[16]:


def getVerbSimilarityPortion(r1, r2):
    vs1 = getVerbLemmas(r1)
    vs2 = getVerbLemmas(r2)
    if len(vs1) < 1 or len(vs2) < 1:
        return 'low'
    num1 = 0
    for n in vs1:
        for m in vs2:
            if n == m:
                num1 = num1 + 1
                break
    if len(vs1) > 0:
        ovlap = num1/len(vs1)
    else:
        ovlap = 0
    if ovlap < 0.2:
        ovlap = 0
    elif 0.2 <= ovlap < 0.4:
        ovlap = 0.3
    elif 0.4 <= ovlap < 0.6:
        ovlap = 0.5
    elif 0.6 <= ovlap < 0.8:
        ovlap = 0.7
    elif 0.8 <= ovlap:
        ovlap = 1
    return ovlap


# In[17]:


def getVSPFeatures(r1, r2):
    res = {'verb_ovlap_1_2' : getVerbSimilarityPortion(r1, r2), 'verb_ovlap_2_1' : getVerbSimilarityPortion(r2, r1)}
    return res


# In[18]:


def getRootLemma(r):
    doc = nlp(r)
    for t in doc:
        if t.dep_ == 'ROOT':
            return t.lemma_
    return 'nothing'


# In[19]:


def getModalityType(r):
    doc = nlp(r)
    mvs = []
    root_lemma = 'nothing'
    for t in doc:
        #print(t.lemma_ + t.tag_ + t.dep_)
        if t.lemma_ == 'be':
            if t.dep_ == 'ROOT':
                return 'fact_be'
        elif t.lemma_ == 'have':
            if t.dep_ == 'ROOT':
                if doc[t.i + 1].lemma_ == 'to':
                    return 'obligatory'
                else:
                    return 'fact_hv'
        elif t.lemma_ == 'need':
            if t.dep_ == 'ROOT':
                return 'fact_need'
        elif t.tag_.startswith('MD'):
            if t.head.dep_ == 'ROOT':
                if t.lemma_ == 'must' or t.lemma_ == 'shall':
                    return 'obligatory'
                elif t.lemma_ == 'can' or t.lemma_ == 'could':
                    return 'ability'
                elif t.lemma_ == 'may':
                    return 'permission'
                elif t.lemma_ == 'should' or t.lemma_ == 'ought':
                    return 'advice'
                elif t.lemma_ == 'will':
                    return 'futurative'
    return 'unknown'
    


# In[20]:


def getModalityFeatures(r1, r2):
    res = {'M_1': getModalityType(r1), 'root_1': getRootLemma(r1), 'M_2': getModalityType(r2), 'root_2': getRootLemma(r2)}
    return res


# In[21]:


def createFBag(r1, r2):
    fbag = getStcSubObjFeatures2(r1, r2)
    fbag.update(getNSPFeatures(r1, r2))
    fbag.update(getVSPFeatures(r1, r2))
    fbag.update(getModalityFeatures(r1, r2))
    return fbag

def createFBagNoOVLAP(r1, r2):
    fbag = getStcSubObjFeatures2(r1, r2)
    fbag.update(getModalityFeatures(r1, r2))
    return fbag

def createFBagNoSUBJOBJ(r1, r2):
    fbag = getNSPFeatures(r1, r2)
    fbag.update(getVSPFeatures(r1, r2))
    fbag.update(getModalityFeatures(r1, r2))
    return fbag

def createFBagNoModal(r1, r2):
    fbag = getStcSubObjFeatures2(r1, r2)
    fbag.update(getNSPFeatures(r1, r2))
    fbag.update(getVSPFeatures(r1, r2))
    return fbag


# In[22]:


#createFBag('Setting the mode to its current value must have no effect on irrigation.', 'Users must be able to set the mode.')
createFBag('Mode must be either automatic or manual.', 'Users must be able to set the mode.')

