# import class LRClassifier
from Classifier import LRClassifier

if __name__ == "__main__":
    # deklarasikan nama kolom dari csv
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    # deklarasikan attribute dari data yang akan diprediksi
    arr = {
        'sepal_length': 4.9,
        'sepal_width': 2.5,
        'petal_length': 4.5,
        'petal_width': 1.7
    }
    # buat instance class LRClassifier
    model = LRClassifier('iris.csv', ',', columns, [], 'class')
    # dapatkan sample data
    print(model.read()) 
    # dapatkan sample data sebanyak 10
    print(model.read(10))
    # lakukan prediksi dari data yang sudah kita buat
    print(model.predict(model.format(arr)))
    # dapatkan laporan akurasi dari dataset
    print(model.getScore(0.2))