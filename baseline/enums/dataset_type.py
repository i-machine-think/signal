from enum import Enum

class DatasetType(Enum):
    Train = 'train'
    Valid = 'validation'
    Test = 'test'

    def __str__(self):
        return self.value