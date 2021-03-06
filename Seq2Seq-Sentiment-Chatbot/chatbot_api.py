import os
from flask import Flask, flash, request, redirect, url_for, jsonify
from tensorflow.python.ops.numpy_ops.np_math_ops import positive
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
    global yaml_filename_1
    global yaml_filename_2
    global positive_sentiment_threshold
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
                print("Saving previous interaction...")
                print("User response was:",query)
                print("Previous Input:", prev_input)
                print("Previous Response:", prev_response)
                save_to_yaml(prev_input, prev_response, yaml_filename_1)
                print("Saving current interaction...")
                print("Input:",prev_response)
                print("Response:",query)
                save_to_yaml(prev_response, query, yaml_filename_2) 
            response = cbot.chat(query=str(query)).replace('\'', '')
            response = response[1:-4]
            print("Chatbot response sentiment: ",sentiment_analyser.predict_sentiment(response))
            prev_input = query
            prev_response = response
            return jsonify({ "Item": response})   
    return '''
    <!doctype html>
    <title>API</title>
    <h1>API Running chatbot Successfully</h1>'''

def save_to_yaml(input, response, yaml_filename):
    print("Saving interaction to yaml file:", yaml_filename,"...")
    with open(yaml_filename,'r') as yamlfile:
        yaml_data = yaml.load(yamlfile, Loader=yaml.FullLoader)

    conv_data = yaml_data['conversations']
    new_data = [input, response]
    conv_data.append(new_data)
    yaml_data['conversations'] = conv_data    

    with open(yaml_filename, 'w') as yamlfile:
        yaml.safe_dump(yaml_data, yamlfile)
    
    print("Saved interaction ",input,":",response,"to yaml file: ",yaml_filename)


if __name__ == "__main__":
    positive_sentiment_threshold = 0.90 #set threshold for positive sentiment
    session_ID = "session_2" #set session ID
    yaml_filename_1= "chatbot_nlp/custom_data_previous/"+session_ID+".yml" #filepath for previous interactions
    yaml_filename_2= "chatbot_nlp/custom_data_current/"+session_ID+".yml" #filepath for current interactions

    #variables to store previous input and response
    prev_input = ""
    prev_response = ""

    #intialize previous interaction file
    with open(yaml_filename_1,'w') as yamlfile:
        data = dict(
            categories = ['If user sentiment is positive, the stored data is the previous interaction'],
            conversations = []
        )
        yaml.safe_dump(data, yamlfile)
    print("Created yaml file:",yaml_filename_1)

    #initialize current interaction file
    with open(yaml_filename_2,'w') as yamlfile:
        data = dict(
            categories = ['If user sentiment is positive, the stored data is the current interaction'],
            conversations = []
        )
        yaml.safe_dump(data, yamlfile)
    print("Created yaml file:",yaml_filename_2)

    #Setup chatbot and sentiment analyser models
    sentiment_analyser = sentiment_analyse.Sentiment_Analyse()
    cbot = S2S_Chatbot.Chatbot(data_directory='chatbot_nlp/data')
    cbot.prep_data()
    model = cbot.make_model()
    cbot.train(load_path='None', save_path='model_base_data.h5', epochs=150)
    cbot.make_inference_models(load_path='model_base_data.h5')
    app.run("0.0.0.0", port=80, debug=False)