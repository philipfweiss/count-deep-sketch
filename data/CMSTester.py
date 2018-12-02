class CMSTester:
    def __init__(self, model):
        self.dataset = None

    def loadDataset(recompute=False):
        raise Exception("Not yet initialized!")

    def trainModel():
        raise Exception("Not yet initialized!")

    def testModel():
        raise Exception("Not yet initialized!")

    def visualizeData():
        raise Exception("Not yet initialized!")



class WikipediaTester(CMSTester):
    def __init__(self, model):
        CMSTester.__init__(self, model)

    """
    Sets self.dataset to the dataset, only recomputes if necessary. 
    """
    def loadDataset(self, recompute=True):
        if recompute:
            self.dataset = "Foo"
        else:
            self.dataset = "bar"

    def trainModel(self):
        print(self.dataset)



wt = WikipediaTester('foo')
wt.loadDataset(False)
wt.trainModel()


## Model,
