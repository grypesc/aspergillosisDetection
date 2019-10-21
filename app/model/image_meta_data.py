class ImageMetaData():
    def __init__(self):
        self.name = ""
        self.diagnosis = "null"
        self.probability = -1
    def __init__(self, nameArg, diagArg, probArg):
        self.name = nameArg
        self.diagnosis = diagArg
        self.probability = probArg
