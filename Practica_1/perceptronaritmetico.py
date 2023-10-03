import numpy as np
from keras.models import Sequential
from keras.src.layers.core import Dense

# Fix random seed for reproducibility.
seed = 7
np.random.seed(seed)

#sumas = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17]]
#multiplicacion = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17]]
#modulo = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17]]
#restas = [[3,0],[5,1],[7,2],[9,3],[11,4],[13,5],[15,6],[17,7],[19,8]]
division = [[50,2],[48,3],[48,4],[45,5],[42,6],[35,7],[32,8],[27,9],[20,10]]

# cargamos las 4 combinaciones de las compuertas 
#training_data = np.array(sumas, "float32")
#training_data = np.array(multiplicacion, "float32")
#training_data = np.array(modulo, "float32")
#training_data = np.array(restas, "float32")
training_data = np.array(division, "float32")
 
# y estos son los resultados que se obtienen, en el mismo orden
#target_data = np.array([[1],[5],[9],[13],[17],[21],[25],[29],[33]], "float32") # suma 10800 {2, 5, 1}
#target_data = np.array([[0],[6],[20],[42],[72],[110],[156],[210],[272]], "float32") # multiplicacion 21900 {2, 5, 1}
#target_data = np.array([[0],[2],[4],[6],[8],[10],[12],[14],[16]], "float32") # modulo 10900 {2, 5, 1}
#target_data = np.array([[3],[4],[5],[6],[7],[8],[9],[10],[11]], "float32") # resta 10900 {2, 5, 1}
target_data = np.array([[25],[16],[12],[9],[7],[5],[4],[3],[2]], "float32") # division 10900 {2, 5, 1}


 
model = Sequential()
model.add(Dense( 2, input_dim=2, kernel_initializer='uniform', activation='softplus'))
model.add(Dense( 5, kernel_initializer='uniform', activation='softplus'))
model.add(Dense( 1, kernel_initializer='uniform', activation='softplus'))
 
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

 
model.fit(training_data, target_data, epochs=10900, batch_size=10)
 
# evaluamos el modelo
scores = model.evaluate(training_data, target_data)
 
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(model.predict(training_data).round())