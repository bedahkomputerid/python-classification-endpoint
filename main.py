# import class LRClassifier
from Classifier import LRClassifier

def get_dataset(name):
    if name == 'iris':
        return {
            'file': 'iris.csv',
            'delimiter': ',',
            'na': [],
            'target': 'class',
            'columns': [
                'sepal_length',
                'sepal_width',
                'petal_length',
                'petal_width',
                'class'
            ],
            'predict': {
                'sepal_length': 5,
                'sepal_width': 2,
                'petal_length': 3.5,
                'petal_width': 1
            }
        }
    elif name == 'zoo':
        return {
            'file': 'zoo.csv',
            'delimiter': ';',
            'na': [],
            'target': 'type',
            'columns': [
                'hair',
                'feathers',
                'eggs',
                'milk',
                'airborne',
                'aquatic',
                'predator',
                'toothed',
                'backbone',
                'breathes',
                'venomous',
                'fins',
                'legs',
                'tail',
                'domestic',
                'catsize',
                'type'
            ],
            'predict': {
                'hair': 0,
                'feathers': 0,
                'eggs': 1,
                'milk': 0,
                'airborne': 0,
                'aquatic': 0,
                'predator': 1,
                'toothed': 1,
                'backbone': 1,
                'breathes': 1,
                'venomous': 0,
                'fins': 0,
                'legs': 4,
                'tail': 1,
                'domestic': 0,
                'catsize': 0
            }
        }

if __name__ == "__main__":
    # deklarasikan nama kolom dari csv
    
    # deklarasikan attribute dari data yang akan diprediksi
    arr = get_dataset('iris')
    # buat instance class LRClassifier
    model = LRClassifier(arr['file'], arr['delimiter'], arr['columns'], arr['na'], arr['target'])
    # dapatkan sample data
    # print(model.read()) 
    # dapatkan sample data sebanyak 10
    # print(model.read(10))
    # lakukan prediksi dari data yang sudah kita buat
    print(model.predict(model.format(arr['predict'])))
    # dapatkan laporan akurasi dari dataset
    # print(model.getScore(0.2))