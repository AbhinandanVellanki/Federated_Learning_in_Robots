import numpy as np
from shutil import copyfile
from tensorflow.keras.models import load_model


def average_weights(modelA, modelB):
    
    copyfile(modelA, 'model_3.h5')
    model = load_model(modelA)
    model2 = load_model(modelB)
    model3 = load_model('model_3.h5')
    weights = model.get_weights()
    weights2 = model2.get_weights()

    print("model",weights[9])
    print("model2",weights2[9])

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
# print(np.subtract(weightsf, weights)[9])
