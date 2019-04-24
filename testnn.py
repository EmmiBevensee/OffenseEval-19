import sklearn.metrics
import pandas as pd

import spacy
from spacy.lang.en.examples import sentences
from spacy.symbols import  LEMMA # POS, TAG, ORTH, add'l if desired
import pickle
import numpy as np

import keras

from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, TensorBoard
from keras import optimizers
from keras.layers import Dense, recurrent, GRU, Bidirectional #Dropout, Flatten, RNN, SimpleRNN,
from keras.layers import Embedding, GlobalMaxPooling1D, LSTM #Conv2D, MaxPooling2D, Conv1D,
from keras import backend as K

from keras.preprocessing import text



#to save model and not have to train again
# from keras.models import load_model model.save('my_model.h5') # creates a HDF5 file 'my_model.h5'
#
# del model # deletes the existing model
#
# # returns a compiled model
# # identical to the previous one
# model = load_model('my_model.h5')
#https://keras.io/getting-started/faq/#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras



#example of poor coding
# 43605	@USER @USER Obama wanted liberals &amp; illegals to move into red states	NOT	NULL	NULL
#56392	@USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER @USER I like my soda like I like my boarders with a lot of ICE.	NOT	NULL	NULL
#3299  17308  @USER Empty headed Ginger Hammer          0       NaN       NaN

#this code won't work bc it's just one column with two possible responses
offense = ['subtask_a']
offense_to_int = {"NOT": 0, "OFF": 1, "NONE": 0}
int_to_offense = {0:"NOT", 1:"OFF"}


#this tokenizes

def preprocessing(data):
    lemma = []
    text = data['tweet']

    nlp = spacy.load('en_core_web_lg') #python -m spacy download en_core_web_lg

    for tweet in text:
        tweet = nlp(tweet)
        lemma.append([])
        for token in tweet:
            # if token != "@USER":
            lemma[-1].append(token.lemma_)

    indexed = []

    max_len = len(max(lemma, key=len))






    #to create vocabulary
    # vocabulary = {}
    # value = 1
    # for tweet in lemma:
    #     for token in tweet:
    #         if token not in vocabulary:
    #             vocabulary[token] = value
    #             value += 1


    #this is if you haven't saved the data and need to re-run it
    if 'subtask_a' in data.columns:
        # pickle.dump(vocabulary, open('vocabulary.pkl', 'wb'))
        #this is if you haven't saved the data and need to re-run it
        pickle.dump(lemma, open('lemma_tokens.pkl', 'wb'))

    # previously created dictionary of vocabulary
    vocabulary = pickle.load(open('vocabulary.pkl', 'rb'))

    #turn tokenized tweets into integers
    for tweet in lemma:
        row = []
        for token in tweet:
            if token in vocabulary: #this ignores new vocabulary from test or dev
                row.append(vocabulary[token]) #is extend better here?
            #possibly have to add an else here to add the word to vocabulary dictionary
            #and then run the rest of it for new words from dev or test
        while len(row) < max_len:
            row.append(0)
        indexed.append(row)

    inputs = np.array(indexed)




    #to differentiate between dev and test
    if 'subtask_a' in data.columns:
        #this is if you haven't saved the data and need to re-run it
        pickle.dump(indexed, open('lemma_indexed.pkl', 'wb'))
        #train outputs
        outputs = data.loc[: , 'subtask_a'].values.copy()
    else:
        # rows = len(data)
        # # zeros = list(np.zeros(rows))
        # data.loc['subtask_a'] = pd.np.zeros(rows, 1)
        # outputs = data.loc[: , 'subtask_a'].values.copy()
        rows = len(data)
        outputs = np.zeros(rows)


    return inputs, outputs, vocabulary




def train_and_predict(train_data: pd.DataFrame,
                      dev_data: pd.DataFrame) -> pd.DataFrame:

    print("...training...espera porfa")
    #this is if you haven't saved the data and need to re-run it
    train_inputs, train_outputs, vocabulary = preprocessing(train_data)

    dev_inputs, dev_outputs, vocabulary = preprocessing(dev_data)


    # train_inputs = np.array(pickle.load(open('lemma_indexed.pkl', 'rb')))
    print("train_inputs")
    print(train_inputs[:10])
    print(train_inputs.shape)

    # train_outputs = train_inputs.loc[:, 'subtask_a'].values.copy() #train_data
    print("train_outputs")
    print(train_outputs[:10])
    print(train_outputs.shape)
    vocabulary = pickle.load(open('vocabulary.pkl', 'rb')) #vocab above but this is the larger vocab from training


    #play with more layers
    rnn = Sequential()
    rnn.add(Embedding(input_dim = len(vocabulary) +1, output_dim = 75))
    rnn.add(Bidirectional(GRU(128, activation='tanh', return_sequences=True)))
    rnn.add(GlobalMaxPooling1D())
    rnn.add(Dense(1, activation='sigmoid'))
    rnn.compile(loss='binary_crossentropy',
             optimizer= 'adam', metrics=['accuracy'])


    #tensorboard --logdir ./logs --host localhost --port 8088  (command script)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1,
                          write_graph=True, write_images=True, write_grads = True)


    rnn.fit(x = train_inputs, y = train_outputs,  epochs = 2, callbacks=[tensorboard]) #callbacks = [EarlyStopping(monitor='loss', patience=1, min_delta=0.01, mode='auto')],



    predictions = np.round(rnn.predict(dev_inputs)).astype(int)
    print("predictions type")
    print(type(predictions))
    print("len of predictions" + str(len(predictions)))
    # print(predictions)

    #what is this doing and is it just overwriting the predictions?
    # dev_predictions = dev_data.copy()

    # dev_predictions[offense] = predictions #this breaks for testdata bc no key 'subtask_a'
    # print("devpredict")
    # print(type(dev_predictions))
    # print(dev_predictions)


    return predictions

def main():

    # reads train and dev data into Pandas data frames
    read_csv_kwargs = dict(sep="\t",
                           converters={e: offense_to_int.get for e in offense})
    train_and_dev = pd.read_csv("offenseval-training-v1.tsv", **read_csv_kwargs)
    len_of_data = len(train_and_dev)

    test_data =  pd.read_csv("testset-taska.tsv",  sep="\t" ) #, **read_csv_kwargs header = 0,
    print("test data")
    print(test_data[:10])
    print("len of test data " + str(len(test_data)))
    #this is for development, turned to notes for test
    #possibly better way to split
#     from sklearn.model_selection import train_test_split
# train, test = train_test_split(df, test_size=0.33, random_state=42)
    # dev_len = int(len_of_data/4)
    # train_data = train_and_dev[:dev_len]
    # print(train_data.head)
    # dev_data = train_and_dev[dev_len:]
    # print(dev_data.head)

    # makes predictions on the dev set
    dev_predictions = train_and_predict(train_and_dev, test_data)
    print("dev_predictions")
    print(dev_predictions[:10])
    print("type: " + str(type(dev_predictions)))
    print("len of dev predictions" + str(len(dev_predictions)))

    # dev_predictions_formatted = dev_predictions


    # The prediction format is a comma-delimited text file with a *.csv extension which
    # should contain two columns: (1) the sample ID, and (2) the sample label.
    #  You can name the file anything you like, so long as it has a .csv extension.
    #  The order of the entries is not important, but there must be one prediction for each ID.
    #   The CSV file should not have a header row.

# saves predictions and prints out multi-label accuracy
    #index = ids?

    # dev_predictions = pd.DataFrame(dev_predictions)
    # dev_predictions = [int_to_offense[int] for int in list(dev_predictions)] #back to labels
    label_dev_predictions =  []
    for int in dev_predictions:
        if int == 0:
            label_dev_predictions.append("NOT")
        else:
            label_dev_predictions.append("OFF")
    test_data['subtask_a'] = label_dev_predictions
    id_and_predictions = test_data.loc[:, ['id', 'subtask_a']]
    id_and_predictions.to_csv("offense_pred.csv", index=False, header= False) #, sep="\t"
    # print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
    #     dev_data[offense], dev_predictions[offense])))




if __name__ == "__main__": #makes it easier to pull these in from elsewhere b/c name == main only here
    main()




#useful bit of shell for problems getting packages
#python -m spacy download en




#to load it all back up
# lemma = pickle.load(open('lemma_tokens.csv', 'rb'))

#might want to create a new vocabulary


#to load it all back up
# vocabulary = pickle.load(open('vocabulary.csv', 'rb'))





    #this is if you haven't saved the data and need to re-run it
# pickle.dump(indexed, open('lemma_indexed.csv', 'wb'))

#to load it all back up
# indexed = pickle.load(open('lemma_indexed.csv', 'rb'))
