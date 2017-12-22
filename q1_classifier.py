
# coding: utf-8

# In[1]:


#Check this later
"""
import argparse, os, sys


parser = argparse.ArgumentParser()

parser.add_argument('-p',
                    help='specify p-value threshold',
                    dest='pValue',
                    action='store',
                    default='0.005'
                    )

parser.add_argument('-f1',
                    help='specify training dataset path',
                    dest='train_dataset',
                    action='store',
                    default=''
                    )

parser.add_argument('-f2',
                    help='specify test dataset path',
                    dest='test_dataset',
                    action='store',
                    default=''
                    )

parser.add_argument('-o',
                    help='specify output file',
                    dest='output_file',
                    action='store',
                    default=''
                    )

parser.add_argument('-t',
                    help='specify decision tree',
                    dest='decision_tree',
                    action='store',
                    default=''
                    )


args = parser.parse_args()

"""
# In[4]:


import pandas as pd
import numpy as np
from scipy.stats import chisquare


# In[5]:


examples = pd.read_csv('train.csv', header=None, sep=" ")


# In[6]:


examples.shape


# In[7]:


examples.head()


# In[9]:


target_attribute = pd.read_csv('train_label.csv',header=None)


# In[10]:


target_attribute.shape


# In[11]:


target_attribute.head()


# In[12]:


examples['output'] = target_attribute[0]


# In[13]:


examples.head()


# In[14]:


#Shall we take attributes in a dataframe?
features = examples.shape[1] - 1
attributes = [i for i in range(features)]


# In[15]:


print ("attributes: ",attributes)


# In[16]:


class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data


# In[17]:


def entropy(attribute, examples):
    
    n = examples[attribute].count()
    uniqueValues = examples[attribute].unique()    
    entropySum = 0
    
    for value in uniqueValues:
        count = (examples[attribute] == value).sum()
        p = float(count)/n
        #Create a temporary dataframe of attribute and output columns:
        tempdf = examples.filter([attribute,'output'],axis=1)
        tempdf = tempdf.loc[(tempdf[attribute]==value)]
        tempdfLen = tempdf['output'].count()
        zeroCount = (tempdf['output'] == 0).sum()
        oneCount = (tempdf['output'] == 1).sum()
        pZero = float(zeroCount)/tempdfLen
        pOne = float(oneCount)/tempdfLen
        
        if pZero == 0:
            entropyZero = 0
        else:
            entropyZero = pZero*(np.log2(pZero))
        
        if pOne == 0:
            entropyOne = 0
        else:
            entropyOne = pOne*(np.log2(pOne))
        
        tEntropy = -(entropyZero + entropyOne )
        
        entropySum += p*float(tEntropy)
    
    return float(entropySum)


# In[18]:


def chooseBestAttribute(examples, attributes):
    #default Values
    minEntropy = 9999999999
    bestAttribute = attributes[0]
    
    for attribute in attributes:
		x = entropy(attribute, examples)
		#print("entropy of ", attribute, " is ",x)
		if  x < minEntropy:
			bestAttribute = attribute
			minEntropy = x
    print("bestAttribute: ", bestAttribute)
    return bestAttribute
            


# In[19]:


def checkChiSquare(examples, bestAttribute, pValue):
    
    fObs = list()
    fExp = list()
    
    zeroCount = (examples['output'] == 0).sum()
    oneCount = (examples['output'] == 1).sum()
    n = examples['output'].count()
    
    zeroRatio = float(zeroCount)/n
    oneRatio = float(oneCount)/n
    
    uniqueValues = examples[bestAttribute].unique()
    
    for value in uniqueValues:
        
        tempdf = examples.filter([bestAttribute,'output'],axis=1)
        tempdf = tempdf.loc[(tempdf[bestAttribute]==value)]
        valueCount = tempdf['output'].count()
        
        observedZeroes = float((tempdf['output'] == 0).sum())
        observedOnes = float((tempdf['output'] == 1).sum())
        
        expectedZeroes = float(zeroRatio)*valueCount
        expectedOnes = float(oneRatio)*valueCount
        
        #not sure check for divide by zero
        fObs.append(observedZeroes)
        fObs.append(observedOnes)
        fExp.append(expectedZeroes)
        fExp.append(expectedOnes)
    
    chiSq, p = chisquare(fObs, fExp)
    
    if p<=pValue:
        return True
    else:
        return False


# In[20]:


def ID3(examples, attributes, pValue):
    
    #Check if all examples are positive:
    if (examples['output'] == 1).sum() == examples['output'].count():
        root = TreeNode('T',children=[-1]*5)
        return root
    
    #Check if all examples are negative:
    if (examples['output'] == 0).sum() == examples['output'].count():
        root = TreeNode('F',children=[-1]*5)
        return root
    
    #If attributes is empty, then select the majority element as output value
    if len(attributes)==0:
        oneCount = 0
        zeroCount = 0
        oneCount = (examples['output'] == 1).sum()
        zeroCount = (examples['output'] == 0).sum()
        if oneCount >= zeroCount:
            root = TreeNode('T',children=[-1]*5)
            return root
        else:
            root = TreeNode('F',children=[-1]*5)
            return root
    
    #Choose the best attribute
    A = chooseBestAttribute(examples, attributes)
    
    if checkChiSquare(examples, A, pValue):
        root = TreeNode(A, children=[-1]*5)
    
        #check if 'A' or A works
        uniqueValues = examples[A].unique()
        attributes.remove(A)

        i=0
        for value in uniqueValues:
            examplesSubset = examples.loc[examples[A] == value]

            if examplesSubset.empty:
                oneCount = (examples['output'] == 1).sum()
                zeroCount = (examples['output'] == 0).sum()
                if oneCount>= zeroCount:
                    root.nodes[i]= TreeNode('T',children[-1]*5)
                else:
                    root.nodes[i] = TreeNode('F', children[-1]*5)
            else:    
                root.nodes[i] = ID3(examplesSubset, attributes, pValue)   
            i+=1
    else:
        #check if a dummy node should be returned or "None" or something else
        return
        
    return root

def BFS(root):
    
    queue = list()
    
    queue.append(root)
    
    while len(queue)>0:
        n = len(queue)
        r = list()
        for i in range(n):
            node = queue[0]
            queue.remove(node)
            if node is not None:
                r.append(node.data)
                for children in node.nodes:
                    if children!=-1:
                        queue.append(children)
        
        print (r)
	print ("queue: ", queue)

# In[22]:


#Check if you need to pass copy(deep=True) for Python 2.x
from copy import deepcopy
examplesBackup = examples.copy(deep=True)
attributesBackup = deepcopy(attributes)


# In[23]:


pValue = 1.0
print("examples: ", examples)
root = ID3(examples, attributes, pValue)


# In[24]:


BFS(root)


# In[25]:


examples = examplesBackup
attributes = attributesBackup


# In[26]:


#root = ID3(examples, attributes, pValue)


# In[27]:


#BFS(root)

