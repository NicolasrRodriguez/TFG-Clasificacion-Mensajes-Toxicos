import pandas as pd 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

from multicolinearidad import atributos_redundantes

datapack = pd.read_csv('Data/pooled_outputs.csv')

caracteristicas = datapack.drop(columns=['class'])

caracteristicas = caracteristicas.drop(labels = atributos_redundantes ,axis= 1)

X_train = scaler.fit_transform(caracteristicas)

clase = datapack['class']

X_train['class'] = datapack['class'].values


cols = [ 0, 1, 2, 3, 4, 5, 8, 9, 14, 15, 19, 20, 22, 23, 24, 28, 30, 34, 35, 37, 38, 40,
    42, 44, 45, 46, 48, 49, 59, 60, 61, 62, 63, 64, 65, 68, 69, 71, 73, 74, 78, 83,
    88, 89, 96, 97, 109, 114, 118, 120, 129, 134, 141, 143, 150, 151, 162, 170, 171,
    172, 174, 176, 178, 181, 182, 187, 190, 191, 193, 195, 203, 204, 205, 213, 217,
    222, 228, 237, 242, 247, 262, 271, 272, 286, 294, 296, 301, 304, 311, 312, 318,
    325, 330, 333, 337, 369, 387, 389, 393, 394, 397, 398, 400, 403, 405, 407, 408,
    409, 416, 432, 435, 436, 446, 448, 449, 456, 469, 475, 477, 490, 492, 494, 496,
    499, 502, 503, 506, 522, 524, 530, 536, 550, 551, 555, 575, 577, 581, 591, 593,
    604, 608, 611, 613, 615, 621, 626, 631, 634, 637, 640, 644, 657, 667, 668, 685,
    688, 689, 696, 697, 700, 701, 704, 711, 713, 716, 727, 739, 758, 764]

#caracteristicas_ =  caracteristicas.drop( columns=  cols, axis=0 )

#print(caracteristicas_.describe())