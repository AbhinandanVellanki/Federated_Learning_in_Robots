import numpy as np
import tensorflow as tf
from tensorflow.keras import layers , activations , models , preprocessing
from tensorflow.keras import preprocessing , utils
import os
import yaml
from gensim.models import Word2Vec
import re

class Chatbot():
    dir_path = ''
    questions = list()
    answers = list()
    answers_with_tags = list()
    tokenizer = preprocessing.text.Tokenizer()
    vocab = []
    VOCAB_SIZE = 0
    maxlen_questions = 0
    maxlen_answers=0

    def __init__(self, data_directory):
        self.dir_path = data_directory
        files_list = os.listdir(self.dir_path + os.sep)
        #initialise questions and answers
        for filepath in files_list:
            stream = open(self.dir_path + os.sep + filepath , 'rb')
            docs = yaml.safe_load(stream)
            conversations = docs['conversations']
            for con in conversations:
                if len( con ) > 2 :
                    self.questions.append(con[0])
                    replies = con[ 1 : ]
                    ans = ''
                    for rep in replies:
                        ans += ' ' + rep
                    self.answers.append( ans )
                elif len( con )> 1:
                    self.questions.append(con[0])
                    self.answers.append(con[1])

        answers_with_tags = list()
        for i in range( len(self.answers ) ):
            if type(self.answers[i] ) == str:
                answers_with_tags.append( self.answers[i] )
            else:
                self.questions.pop( i )

        self.answers = list()
        for i in range( len( answers_with_tags ) ) :
            self.answers.append( '<START> ' + answers_with_tags[i] + ' <END>' )

        self.tokenizer.fit_on_texts( self.questions + self.answers )
        self.VOCAB_SIZE = len( self.tokenizer.word_index )+1
        print( 'VOCAB SIZE : {}'.format( self.VOCAB_SIZE ))

    def tokenize(self, sentences ):
        tokens_list = []
        vocabulary = []
        for sentence in sentences:
            sentence = sentence.lower()
            sentence = re.sub( '[^a-zA-Z]', ' ', sentence )
            tokens = sentence.split()
            vocabulary += tokens
            tokens_list.append( tokens )
        return tokens_list , vocabulary

    def prep_data(self):
        for word in self.tokenizer.word_index:
            self.vocab.append(word)
        
        #save base model vocabulary
        with open('model_vocab.txt', 'w') as f:
            for item in self.vocab:
                f.write("%s\n" % item)

        tokenized_questions = self.tokenizer.texts_to_sequences(self.questions )
        self.maxlen_questions = max( [ len(x) for x in tokenized_questions ] )
        padded_questions = preprocessing.sequence.pad_sequences( tokenized_questions , maxlen=self.maxlen_questions , padding='post' )
        self.encoder_input_data = np.array( padded_questions )
        print("Max length of questions: ", self.maxlen_questions)
        print("Encoder input data shape: ",self.encoder_input_data.shape )

        # decoder_input_data
        tokenized_answers = self.tokenizer.texts_to_sequences(self.answers )
        self.maxlen_answers = max( [ len(x) for x in tokenized_answers ] )
        padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=self.maxlen_answers , padding='post' )
        self.decoder_input_data = np.array( padded_answers )
        print("Max length of answers: ", self.maxlen_answers)
        print("Decoder input data shape: ", self.decoder_input_data.shape)

        # decoder_output_data
        tokenized_answers = self.tokenizer.texts_to_sequences(self.answers )
        for i in range(len(tokenized_answers)) :
            tokenized_answers[i] = tokenized_answers[i][1:]
        padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=self.maxlen_answers , padding='post' )
        onehot_answers = utils.to_categorical( padded_answers ,self.VOCAB_SIZE )
        self.decoder_output_data = np.array( onehot_answers )
        print("Decoder output data shape: ",self.decoder_output_data.shape )
    
    def make_model(self):
        self.encoder_inputs = tf.keras.layers.Input(shape=( self.maxlen_questions , ))
        self.encoder_embedding = tf.keras.layers.Embedding( self.VOCAB_SIZE, 200 , mask_zero=True ) (self.encoder_inputs)
        encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( self.encoder_embedding )
        self.encoder_states = [ state_h , state_c ]

        self.decoder_inputs = tf.keras.layers.Input(shape=( self.maxlen_answers ,  ))
        self.decoder_embedding = tf.keras.layers.Embedding( self.VOCAB_SIZE, 200 , mask_zero=True) (self.decoder_inputs)
        self.decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
        self.decoder_outputs , _ , _ = self.decoder_lstm ( self.decoder_embedding , initial_state=self.encoder_states )
        self.decoder_dense = tf.keras.layers.Dense( self.VOCAB_SIZE , activation=tf.keras.activations.softmax ) 
        output = self.decoder_dense ( self.decoder_outputs )

        self.model = tf.keras.models.Model([self.encoder_inputs, self.decoder_inputs], output )
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

        self.model.summary()
        return self.model
    
    def train(self, save_path):
        self.model.fit([self.encoder_input_data , self.decoder_input_data], self.decoder_output_data, batch_size=50, epochs=150 ) 
        self.model.save(save_path)

    def make_inference_models(self, load_path):
        self.model.load_weights(str(load_path))
        self.encoder_model = tf.keras.models.Model(self.encoder_inputs, self.encoder_states)
        
        decoder_state_input_h = tf.keras.layers.Input(shape=( 200 ,))
        decoder_state_input_c = tf.keras.layers.Input(shape=( 200 ,))
        
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_outputs, state_h, state_c = self.decoder_lstm(self.decoder_embedding , initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        self.decoder_model = tf.keras.models.Model([self.decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)
        
    def str_to_tokens( self,sentence : str ):
        words = sentence.lower().split()
        tokens_list = list()
        for word in words:
            try:
                tokens_list.append( self.tokenizer.word_index[ word ] ) 
            except KeyError as k:
                return k
        return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=self.maxlen_questions , padding='post')


    def chat(self, query):
        model_input = self.str_to_tokens(query)
        if str(type(model_input)) == '<class \'KeyError\'>':
            return " I do not know what "+str(model_input)+" means, can you rephrase your sentence end"
        states_values = self.encoder_model.predict( model_input )
        empty_target_seq = np.zeros( ( 1 , 1 ) )
        empty_target_seq[0, 0] = self.tokenizer.word_index['start']
        stop_condition = False
        response = ''
        while not stop_condition :
            dec_outputs , h , c = self.decoder_model.predict([ empty_target_seq ] + states_values )
            sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
            sampled_word = None
            for word , index in self.tokenizer.word_index.items() :
                if sampled_word_index == index :
                    response += ' {}'.format( word )
                    sampled_word = word

            if sampled_word == 'end' or len(response.split()) > self.maxlen_answers:
                stop_condition = True
                
            empty_target_seq = np.zeros( ( 1 , 1 ) )  
            empty_target_seq[ 0 , 0 ] = sampled_word_index
            states_values = [ h , c ] 
        return response
    
    def main(self):
        self.prep_data()
        model = self.make_model()
        #self.train(save_path='new_model.h5')
        self.make_inference_models(load_path='new_model.h5')
        query = ''
        while True:
            query = str(input("Enter Question: "))
            if query == 'exit':
                print("Goodbye!")
                break
            response = self.chat(query)
            print(response)

if __name__ == '__main__':
    test_chatbot = Chatbot(data_directory='chatbot_nlp/data')
    test_chatbot.main()