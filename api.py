import os
from flask import Flask, flash, request, redirect, url_for, jsonify
import S2S_Chatbot
import sentiment_analyse
import yaml

os.environ['KMP_DUPLICATE_LIB_OK']='True' #Uncomment on Mac OS
UPLOAD_FOLDER = './uploads/'
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_input():
    global prev_input
    global prev_response
    global yaml_filename
    if request.method == 'POST':
        # check if the post request has the input part
        if not request.json:
            print("no json")
            return redirect(request.url)
        if 'input' not in request.json:
            flash('No input')
            return redirect(request.url)
        query = request.json['input']
        # if user does not select input
        if query == '':
            flash('No selected input')
            return jsonify({ "Item": "why won't you say something"} )
        else:
            if prev_input != "" and prev_response != "" and sentiment_analyser.predict_sentiment(query) > positive_sentiment_threshold:
               print("User sentiment is positive!:", sentiment_analyser.predict_sentiment(query))
               print("User response was:",query)
               print("Previous Input:", prev_input)
               print("Previous Response:", prev_response)
               save_to_yaml(prev_input, prev_response, yaml_filename)
            if len(query) > cbot.maxlen_questions:
                response = 'Sorry! I cannot process such a long sentence, please say something shorter'
            else:
                response = cbot.chat(query=str(query)).replace('\'', '')
                response = response[1:-4]
                print("Chatbot response sentiment: ",sentiment_analyser.predict_sentiment(response))
            prev_input = query
            prev_response = response
            return jsonify({ "Item": response} )    
    return '''
    <!doctype html>
    <title>API</title>
    <h1>API Running chatbot Successfully</h1>'''

def save_to_yaml(input, response, yaml_filename):
    print("Saving interaction to yaml file:", yaml_filename)
    with open(yaml_filename,'r') as yamlfile:
        yaml_data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print(yaml_data)

    conv_data = yaml_data['conversations']
    new_data = [input, response]
    conv_data.append(new_data)
    yaml_data['conversations'] = conv_data    

    with open(yaml_filename, 'w') as yamlfile:
        yaml.safe_dump(yaml_data, yamlfile)
    
    print("Saved to yaml file: ",yaml_filename)


if __name__ == "__main__":
    positive_sentiment_threshold = 0.80 #set threshold for positive sentiment
    yaml_filename = "custom.yml"

    #variables to store previous input and response
    prev_input = ""
    prev_response = ""

    with open(yaml_filename,'w') as yamlfile:
        #cur_yaml = yaml.load(yamlfile, Loader=yaml.FullLoader)
        data = dict(
            categories = ['custom conversation data'],
            conversations = []
        )
        yaml.safe_dump(data, yamlfile)
    
    print("Created yaml file:",yaml_filename)

    #Setup chatbot and sentiment analyser models
    sentiment_analyser = sentiment_analyse.Sentiment_Analyse()
    cbot = S2S_Chatbot.Chatbot(data_directory='chatbot_nlp/data')
    cbot.prep_data()
    model = cbot.make_model()
    #cbot.train(save_path='model_2.h5')
    cbot.make_inference_models(load_path='modelf.h5')
    app.run("0.0.0.0", port=80, debug=False)