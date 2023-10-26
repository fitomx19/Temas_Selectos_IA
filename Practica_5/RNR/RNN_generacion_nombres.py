import numpy as np
import re
from keras.layers import Input, Dense, SimpleRNN
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import backend as K

# Unificar y ordenar todos los nombres
archivos_nombres = ['./RNR/RNN_county.txt']
todos_los_nombres = []
for archivo in archivos_nombres:
    with open(archivo, 'r') as f:
        todos_los_nombres.extend(f.readlines())
todos_los_nombres = sorted(todos_los_nombres)

with open('todos_los_nombres.txt', 'w') as f:
    f.writelines(todos_los_nombres)

# Leer el set de datos
nombres = open('todos_los_nombres.txt', 'r').read().lower()

# Crear diccionario
alfabeto = list(set(nombres))
tam_datos, tam_alfabeto = len(nombres), len(alfabeto)
car_a_ind = {car: ind for ind, car in enumerate(sorted(alfabeto))}
ind_a_car = {ind: car for ind, car in enumerate(sorted(alfabeto))}

# Inicializar variables
n_a = 25  # Número de unidades en la capa oculta

# Definir modelo
entrada = Input(shape=(None, tam_alfabeto))
a0 = Input(shape=(n_a,))
celda_recurrente = SimpleRNN(n_a, activation='tanh', return_state=True)
capa_salida = Dense(tam_alfabeto, activation='softmax')

salida = []
hs, _ = celda_recurrente(entrada, initial_state=a0)
salida.append(capa_salida(hs))
modelo = Model([entrada, a0], salida)
opt = SGD(learning_rate=0.0005)
modelo.compile(optimizer=opt, loss='categorical_crossentropy')

# Generar ejemplos de entrenamiento
def train_generator():
    while True:
        ejemplo = todos_los_nombres[np.random.randint(0, len(todos_los_nombres))].strip().lower()
        X = [None] + [car_a_ind[c] for c in ejemplo]
        Y = X[1:] + [car_a_ind['\n']]
        x = np.zeros((len(X), 1, tam_alfabeto))
        onehot = to_categorical(X[1:], tam_alfabeto).reshape(len(X) - 1, 1, tam_alfabeto)
        x[1:, :, :] = onehot
        y = to_categorical(Y, tam_alfabeto).reshape(len(X), tam_alfabeto)
        a = np.zeros((len(X), n_a))
        yield [x, a], y

# Entrenar el modelo
BATCH_SIZE = 80
NITS = 10000
for j in range(NITS):
    historia = modelo.fit(train_generator(), steps_per_epoch=BATCH_SIZE, epochs=1, verbose=0)
    if j % 1000 == 0:
        print(f'Iteración: {j}, Error: {historia.history["loss"][0]}')

# Generar nombres con el modelo entrenado
def generar_nombre():
    x = np.zeros((1, 1, tam_alfabeto))
    a = np.zeros((1, n_a))
    nombre_generado = ''
    fin_linea = '\n'
    car = -1
    contador = 0

    while (car != fin_linea and contador != 50):
        a, _ = celda_recurrente(K.constant(x), initial_state=K.constant(a))
        y = capa_salida(a)
        prediccion = K.eval(y)
        ix = np.random.choice(list(range(tam_alfabeto)), p=prediccion.ravel())
        car = ind_a_car[ix]
        nombre_generado += car
        x = to_categorical(ix, tam_alfabeto).reshape(1, 1, tam_alfabeto)
        a = K.eval(a)
        contador += 1

        if contador == 50:
            nombre_generado += '\n'

    if len(nombre_generado.strip()) <= 4:
        return None

    if re.search(r"[aeiou]{3,}|[bcdfghjklmnpqrstvwxyz]{3,}", nombre_generado):
        return None

    return nombre_generado

for i in range(100):
    nombre = None
    while nombre is None:
        nombre = generar_nombre()
    print(nombre)
