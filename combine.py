import numpy as np
from shutil import copyfile
from tensorflow.keras.models import load_model
import S2S_Chatbot


cbot = S2S_Chatbot.Chatbot(data_directory='chatbot_nlp/data')
cbot.prep_data()
model3 = cbot.make_model()
model = load_model('model_1.h5')
model2 = load_model('model_2.h5')

weights = model.get_weights()
weights2 = model2.get_weights()

print("model",weights[9])
print("model2",weights2[9])
print("========================================================")
weightstemp = np.add(weights, weights2)
weightsf = []
# print(out_arr)
#out = out/2
#print(out)
for a, b in zip(weights, weights2):
    if type(a) == type(b) and a.size == b.size:
        c = (a+b)/2
        weightsf.append(c)
    else:
        print("error")

model3.set_weights(weightsf)
model3.save('modelf.h5')
print("new model saved!!")
#model4 = load_model('modelf.h5')
# weights3 = np.array(model4.get_weights())
#difference = np.subtract(weightsf, weightstemp)
#print(difference[8])
#print(weights.shape, weights2.shape, weightsf.shape)
# print("----------------------------------------")
# #print(np.subtract(difference, weightsf))
# print(weightsf)
# print("-------------------------------------------------------")
print(np.subtract(weightsf, weights)[9])
