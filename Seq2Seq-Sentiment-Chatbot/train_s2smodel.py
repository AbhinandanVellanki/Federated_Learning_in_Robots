import os
import shutil
import S2S_Chatbot

def move_data(source, destination):
    path = '/Users/apple/Documents/Senior Design Project/Seq2Seq-Sentiment-Chatbot'
    print("Moving yml files in", source, "to", destination,"...")
    for file in os.listdir(source):
        if file.endswith('.yml'):
            print("Moving",file)
            shutil.move(os.path.join(path,source,file), os.path.join(destination,file))
    print("Done moving!")
    return destination

def train_model(data_directory, save_model_name, load_model_name, epochs):
    print("Training model with",data_directory,"...")
    cbot = S2S_Chatbot.Chatbot(data_directory=data_directory)
    cbot.prep_data()
    model = cbot.make_model()
    cbot.train(load_path=load_model_name, save_path=save_model_name, epochs=epochs)
    #cbot.make_inference_models(load_path='model_base_data2.h5')

def main(source, destination, load_model_name, save_model_name, epochs):
    data_directory = move_data(source, destination)
    train_model(data_directory=data_directory, load_model_name=load_model_name, save_model_name=save_model_name, epochs=epochs)

if __name__=='__main__':
    main(source='chatbot_nlp/custom_data_previous/', destination='chatbot_nlp/data/', save_model_name='model_retrained_data', load_model_name='model_base_data.h5', epochs=100)