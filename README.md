### Assignment Description

Implement a chatbot with a rule-based and similarity-based component, image classification component, toy world reasoning system, sequence to sequence neural network and reinforcement learning demonstration. Use a chosen chatbot topic with access to related data sets.


# Introduction

This is a console based chatbot with a football related domain. It helps answer user question relating to Arsenal and football and provides other features such as image classification of player, user-input storage, previous match prediction, and player evolution simulation which are all discussed below. 

The PSEL standards were met in the development of this chatbot. The mission and vision standard for this chatbot was met by ensuring that the chatbot domain did not stray from the main topic in order to provide the appropriate final product.

## Rule Based and Similarity Based Component 

The rule based and similarity-based component consists of pre-processing functions such as punctuation removals, text lemmatization, and word categorization within the main chatbot program. It also includes a sentence tokenizer for the user input, in order to implement and utilise the tf-idf and cosine similarity model which helps provide a suitable response to the user input from a pre-defined text document with an appropriate corpus to the chatbot domain. The chatbot also has appropriate aiml flags to be triggered in order to generate a suitable response to user greetings, farewells etc. Available in the aiml document.

#### Related Files

"std-startup.xml" the program learns the std-startup file and then loads the Arsenal.aiml file.

"aiml/Arsenal.aiml" the program loads the aiml file and records the time it takes to learn it, and prints it to the user.

"question_answer_pairs.txt" retrieves suitable responses to the user based on the corpus in this file.

## Image Classification Component

The image classification component of the chatbot is used to predict whether an arsenal player is a male or a female player and is implemented using a multi-layered convolutional network in a sequential model. It contains 3 convolution2D layers to extract the features of the images, with 64 filters on the last layer. 3 relu activation layers for rectified linear activation. 3 maxpooling2D layers to reduce spatial volume of the images after convolution. It then flattens the data, as convolution is 2D and the dense layer requires a 1D data set, before passing it to the 64 node dense layer to connect the neural network. It then contains another relu activation layer, as well as, a dropout layer with an input fraction rate of 0.5 at each update during training. Finally, the output layers are a 1 node dense layer along with a sigmoid activation layer for 1 / (1 + exp(-x)).

The model is then trained, saved and then loaded into the chatbot using a function. The testing generator is then created in another function using the path for the images and returns the generator which is then used in another function that takes the generator as a parameter and predicts the images in the path mentioned.

#### Related Files

"dataSet/testingData/images/.." includes the images used for predictions.

"imageClassification.py" is the code used for training the network on a dataset to make predictions. The result is stored in a .h5 file.

"myModel.h5" is the pre-trained model used for predictions.

## Toy World Reasoning System

The toy world reasoning system allows the user to input which players will play on which day, which in turn allows the chatbot to store when these players are playing and answer questions to the user later on regarding which players are playing on which day as well as whether a specific player is going to play on a specific day.
The toy world reasoning system is implemented using the natural language toolkit and first order logic. Using alternative simple grammar, along with transitive verbs and quantifiers, Arsenal player names were added to the grammar text document along with weekdays to allow the user to specify the players and days. Transitive verbs such as ‘plays on’ and is ‘playing on’ were also added to accommodate the storage of this information. 
Categories were added with wildcards in the aiml file with <star> tags to match those wildcards, in order to facilitate the appropriate responses to the user questions specified previously. If those patterns are matched in the user input, then a valuation is made on the input string and the nltk model generates the appropriate response to the user.
  
#### Related Files
  
"arsenal-sem.fcfg" includes the set of grammar rules used in this component.

"aiml/Arsenal.aiml" includes the wildcards used in this component (added in addition to the aiml rules from the rule based and similarity based component).


## Transformer Network Component

The transformer network component is used to make predictions on which game has been played on which date. The user provide a date as an input to the chatbot and the chatbot processes the date and returns the 2 teams that played on that date and the result of that game. The transformer network is implemented by first loading an excel spreadsheet with premier league games dating back to early 2000 and all the way to 2018. It then, without pre-processing the input as it is already concise, builds the tokenizer using tfds for both the question and answer pairs and then defines the start and end token. Following that the model tokenizes, filters and then pads the inputs. It then constructs the tensorflow data dataset as the input pipeline before implementing the scaled dot-product attention layer which takes 3 values, query, key and value. Value being the one that decides the amount of importance given to a query. The model then implements a multi-head attention class with model subclassing where each block in the class is put through dense layers and split into multiple heads. Following that the positional encoding class is implemented (to give the model information about the relative position of the input in the sentence) before the encoders, and encoder layers along with their decoders and decoder layer counterparts are implemented (which all consist of sublayers), as well as the transformer. Which includes an encoder, decoder and a final linear layer.

The model is then trained and the weights of the model are saved to be loaded into the main chatbot along with the architecture of the model to allow the user to input dates for predictions to be made.

I must note however that this does not function as intended. Due to there being multiple games on the same dates and some teams having similar names it seems to not operate as intended. Example is shown below of the lack of accuracy.

#### Related Files

"english-premier-league/final_dataset.csv" dataset used to train the model. Dataset from https://www.kaggle.com/saife245/english-premier-league

"transformerGameStats.py" transformer network code that is used to train a model and save the weights to a .h5 file.

"transformerWeights.h5" weights that are loaded and used to build architecture model. 

## Reinforcement Learning Extension

The reinforcement learning extension part of the chatbot, allows the user to choose a player, provide their opinion-based skill rating, as well as how far they would like their potential to be met. Based on that the chatbot provides a simulation of the evolution of the player as they are trained in their time at the club and allows the user to view the final result and how many weeks it took to achieve that result.

The reinforcement learning extension is implemented using the deap library and uses the one max problem to simulate the evolution of a player. It utilises a function which takes 3 parameters, the name of the player, the potential the user seeks, and the player skill. The function generates the attributes of the player such as attacking, defending, and initializes structures which include skills such as shooting, tackling, etc. It then registers the list of players and defines the goal operation (to develop the player), the crossover operator, and the mutation probability (how a player’s skills can go up or down depending on what he is training). The staff pool is then defined to be at 250 (the staff pool is the entities that will help train the player, it includes coaches, players, club personnel etc). The player list is then evaluated before the evolution of the player commences. The crossover and mutation are then applied on the acquired skills as the player develops their skills. After the evolution ends the stats are then printed for the user to display.

## MainChatbot.py


