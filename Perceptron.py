import sys
import getopt
import os
import math
import operator
import random 

class Perceptron:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """Perceptron initialization"""
    #in case you found removing stop words helps.
    #TODO:try removing stop words 
    self.stopList = set(self.readFile('../data/english.stop'))
    self.FILTER_STOP_WORDS = False
    self.numFolds = 10
 
    self.vocab_list = []        #unique words of the vocabulary over all training samples 
    self.word_index_lookup ={}  #dictionary assigning a simple integer for each of the vocabulary word. 'key':vocab word, 'value': integer
    
    #weights and bias
    self.wt_vec_w0= []
    self.wt_vec_wavg= []
    self.wt_vec_final =[]
    self.b0 = 0.0 
    self.bavg =0.0
    self.bfinal= 0.0
    self.c =1 
 #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Perceptron classifier with
  # the best set of features you found through your experiments with Naive Bayes.

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
 
    # Write code here
    x_word_count ={}
    result= self.bfinal
    
    #constructing x vector 
    for word in words:
        if word in x_word_count:
            x_word_count[word] +=1 
        else:
            x_word_count[word] =1 
           

    for ele in x_word_count:
        if ele in self.word_index_lookup:
            index = self.word_index_lookup[ele]
            result+= self.wt_vec_final[index]*x_word_count[ele]
    
    #print(result)
    if result >0:
        return 'pos'
    else:
        return 'neg'

  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Perceptron class.
     * Returns nothing
    """
    
    x_word_count ={}
    result= self.b0
   
    #constructing x vector 
    for word in words:
        if word in x_word_count:
            x_word_count[word] +=1 
        else:
            x_word_count[word] =1 

    for ele in x_word_count:
        index= self.word_index_lookup[ele]
        result += self.wt_vec_w0[index]*x_word_count[ele]
            
    yn = 1 if klass is 'pos' else -1  
    
    if result*yn <=0:
        for ele in x_word_count:
            index= self.word_index_lookup[ele]
            self.wt_vec_w0[index] += yn*x_word_count[ele]
            self.wt_vec_wavg[index] += self.c*yn*x_word_count[ele]
        self.b0 += yn 
        self.bavg+= self.c*yn 
        
     
  def train(self, split, iterations):
      """
      * TODO 
      * iterates through data examples
      * TODO 
      * use weight averages instead of final iteration weights
      """
      #added all training samples words to vocabulary
      for example in split.train:
        words = example.words
        if self.FILTER_STOP_WORDS:
            words =  self.filterStopWords(words)
        
        self.vocab_list += words 
      
      #To remove duplicates from vocab list
      self.vocab_list = list(set(self.vocab_list))       
      l = len(self.vocab_list)

      for i,ele in enumerate(self.vocab_list):
        self.word_index_lookup[ele] = i 
      
      self.wt_vec_w0 = [0.0]*l 
      self.wt_vec_wavg= [0.0]*l
      
      random.shuffle(split.train)  
      for ite in range(iterations):
          for example in split.train:
              words = example.words
              if self.FILTER_STOP_WORDS:
                words =  self.filterStopWords(words)
              self.addExample(example.klass, words)     #this step updates the weights based on the new example  
              self.c +=1 
      
      self.wt_vec_final=[0.0]*l 
      
      for i in range(l):
        self.wt_vec_final[i]  = self.wt_vec_w0[i] - (self.wt_vec_wavg[i])/self.c
      
      self.bfinal= self.b0 - (self.bavg)/self.c 

  # END TODO (Modify code beyond here with caution)
  #############################################################################
   
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits
  
  
  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def test10Fold(args):
  pt = Perceptron()
  
  iterations = int(args[1])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = Perceptron()
    accuracy = 0.0
    classifier.train(split,iterations)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy
    
    
def classifyDir(trainDir, testDir,iter):
  classifier = Perceptron()
  trainSplit = classifier.trainSplit(trainDir)
  iterations = int(iter)
  classifier.train(trainSplit,iterations)
  testSplit = classifier.trainSplit(testDir)
  #testFile = classifier.readFile(testFilePath)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print '[INFO]\tAccuracy: %f' % accuracy
    
def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  
  if len(args) == 3:
    classifyDir(args[0], args[1], args[2])
  elif len(args) == 2:
    test10Fold(args)

if __name__ == "__main__":
    main()
