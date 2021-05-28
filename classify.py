import numpy as np,numpy.random
from numpy.core.fromnumeric import size

def classify(paragraph):
    classes = {"informative":"","affirmative":"","negative":"","financial":"","representation":"",
    "event_of_default":""}
    paragraph_classifier = []
    for i in range(len(paragraph)):
        random = list(numpy.random.dirichlet(np.ones(6),size =1))
        j = 0
        for clas in classes:
            classes[clas] = random[0][j]
            j = j+1
        paragraph_classifier.append(classes)
    
    return paragraph_classifier

    

