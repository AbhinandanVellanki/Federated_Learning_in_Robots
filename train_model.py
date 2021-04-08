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

def train_model(data_directory, model_name, epochs):
    print("Training model with",data_directory,"...")
    cbot = S2S_Chatbot.Chatbot(data_directory=data_directory)
    cbot.prep_data()
    model = cbot.make_model()
    cbot.train(save_path=model_name, epochs=epochs)
    #cbot.make_inference_models(load_path='model_base_data2.h5')

def main(source, destination, model_name, epochs):
    data_directory = move_data(source, destination)
    train_model(data_directory, model_name, epochs)

if __name__=='__main__':
    main(source='chatbot_nlp/custom_data_current/', destination='chatbot_nlp/data/', model_name='model_retrained_data.h5', epochs=100)


