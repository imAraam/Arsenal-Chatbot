#tf-idf model and cosine similarity model with the help of
##https://medium.com/@randerson112358/build-your-own-ai-chat-bot-using-python-machine-learning-682ddd8acc29
##code structure for toy world reasoning feature used from AI lab week 15
import aiml
import timeit
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer

from transformerGameStats import * #import transformer model 

#deap library one max problem implementation from
#https://github.com/DEAP/deap/blob/05518db9d9cfe040cbd8e912ef7a1b3d6e53699b/examples/ga/onemax.py#L159
import random
from deap import base, creator, tools

warnings.filterwarnings('ignore') #ignore warnings


timeBegin = timeit.default_timer() #auxiliary feature to measure time taken to train

kernel = aiml.Kernel()
kernel.learn("std-startup.xml")
kernel.respond("LOAD AIML B")


#remove punctuation
##splits str using regular expression
def PunctRemove(data):
    remove_punct = RegexpTokenizer(r'\w+')
    stringList = " ".join(data)
    noPunct = remove_punct.tokenize(stringList)
    return noPunct

#find correct word category in order to lemmatize word 
def WordType(w):
    tag = nltk.pos_tag([w])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

#function to return lemmatized lower case words
def LemData(data):

    data = data.lower()
    PunctRemove(data)

    lem = WordNetLemmatizer()

    data = [lem.lemmatize(w, WordType(w)) for w in nltk.word_tokenize(data)]
    return data


#main function used to respond to the user
##takes user input as parameter
def Response(userInput):


    userInput = userInput.lower() #change user input to lower case input
    botOutput = ''
    
    with open('question_answer_pairs.txt') as file:
        
        reader = file.read().lower()
        file.close()
    #reader is used to preprocess the corpus and find a matching answer

    
    with open('question_answer_pairs.txt') as file2:
        reader2 = file2.read()
        file2.close()
    #reader2 is used to compare the matching answer and provide the user
    ##with the matching answer that isn't in lower case


    #tokenize document body
    ##bag of words
    tokens = nltk.sent_tokenize(reader)

    tokens.append(userInput)
    


    #implement tf-idf and cosine similarity model

    #create tf object
    tf = TfidfVectorizer(tokenizer = LemData, stop_words='english')
    #transform to matrix
    tfidf_responses = tf.fit_transform(tokens)

    #store similarity score in values
    values = cosine_similarity(tfidf_responses[-1], tfidf_responses)

    #retrieve most compatible response in index
    index = values.argsort()[0][-2]
    
    flat = values.flatten()
    flat.sort() #sort values in asc
    
    score = flat[-2] #most similar score

    #if similarity score is 0 that means bot could not find anything appropriate to respond with
    if(score == 0):
        botOutput = botOutput+"Sorry, I don't quite understand. Could you rephrase?"
    else:
        #tokenize the reader2 corpus to be able to locate matching answer with list index
        botResponseList = nltk.sent_tokenize(reader2)

        #responds with appropriate answer in terms of index
        botOutput = botResponseList[index]
        

    return botOutput



def loadModel(modelName): #load pre-trained .h5 model file
    model = load_model(modelName)
    return model



def createGenerator(testingDataPath):   

    testGen = ImageDataGenerator(rescale = 1./255)

    #test set gen
    testingGenerator = testGen.flow_from_directory(
         testingDataPath,
         target_size=(150, 150),
         color_mode = "rgb",
         shuffle = False,
         batch_size = 32,
         class_mode = 'binary')
    return testingGenerator



def predictImage(generator):

    model = loadModel('myModel.h5')

    testFilenames = generator.filenames
    testingSamples = len(testFilenames)

    generator.reset()
    prediction = model.predict_generator (generator, steps = len(generator), verbose = 0)
    predictedClass = np.round (prediction)

    count = 0
    x = 0
    print ("\n")
    while count < testingSamples:
         if predictedClass[x] == 1:
              print ('Picture number', x +1, 'is a female player.') #+1 as numbering begins from 0
         elif predictedClass[x] == 0:
              print('Picture number', x +1, 'is a male player.')
         else:
              print ("I'm not quite sure..")
         count += 1
         x += 1
         if count == testingSamples:
              break

def getTransformerModel(): #building model architecture and loading weights
     learning_rate = CustomSchedule(D_MODEL)
     optimizer = tf.keras.optimizers.Adam(
          learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
     
     model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
     model.load_weights('transformerWeights.h5')


#-------------------------------------------------------------------------------------------------------------
#lreinforcement learning using deap
#-------------------------------------------------------------------------------------------------------------

def potentialSimulator(playerName, potentialSeeked, playerSkill): #One Max Problem used to simulate
#the time taken for an Arsenal player to reach certain potentials 

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator 
    toolbox.register("attr_bool", random.randint, 0, 1)#attributes such as attacking, defending. Depends on type of player such as forward, left back.

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
        toolbox.attr_bool, 100)#skills such as shooting, tackling etc

    #population is list of players
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #the goal to be maximized (skills)
    def evalOneMax(individual):
        return sum(individual),

    toolbox.register("evaluate", evalOneMax)#goal operation
    toolbox.register("mate", tools.cxTwoPoint)#crossover operator
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)#probability to flip skills by 'indpb'

    toolbox.register("select", tools.selTournament, tournsize=3)#operator to select the players with best skills to remain in the team
    #and mentor other players to be nurtured with similar/better skills than them
    #tournsize is number of players to remain in the team
    
    random.seed(64)

    #initial staff pool for training represented as integers
    pop = toolbox.population(n=250)


    # CXPB  is the probability with which two skills
    #       are crossed
    # MUTPB is the probability for mutating a skill
    CXPB, MUTPB = playerSkill * 0.1, 0.2
    
    print("     Start of evolution: \n")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("Mentored by %i club officials" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while max(fits) < potentialSeeked and g < 1000:
        # A new generation
        g = g + 1
        
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the acquired skills
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individual skills attributes with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the new skills
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate a skill with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the skills with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        
        # The population is entirely replaced by the new
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        if g == 1:
            print("\n-- Week %i --" % g)
            print(" ",playerName, "was mentored %i times during this week" % len(invalid_ind))
            print("  Min potential of", playerName, "%s" % min(fits))
            print("  Max potential of", playerName, " %s" % max(fits))

        if max(fits) == potentialSeeked:
            print("\n-- Week %i --" % g)
            print(" ",playerName, "was mentored %i times during this week" % len(invalid_ind))
            print("  Min potential of", playerName, "%s" % min(fits))
            print("  Max potential of", playerName, " %s\n" % max(fits))
        
    
    print("-- End of successful simulation --")
    print("-- Weeks taken to reach full potential: ", g, "--\n")



    

#aiml triggers
greetings = ['hi', 'hello', 'hi there', 'hey', 'howdy', 'how are you?']
farewells = ['bye', 'take care', 'goodbye', 'see you later']
thanks = ['thanks', 'thank you', 'that is helpful', 'cheers']
arsenal = ['do you support arsenal?', 'no', 'yes']


#initialise first order logic based agent
    #v contains players, different types of games Arsenal competes in, and days of the week
v = """
lacazette => l
aubameyang => a
bernd => b
monday => m
tuesday => t
wednesday => w
thursday => h
friday => f
saturday => s
sunday => su
plays_on => { (l, m), (a, t), (b, w), (l, h), (a, f), (b, s)}
"""
val = nltk.Valuation.fromstring(v)
grammarFile = 'arsenal-sem.fcfg'
objectCounter = 0


flag = True

getTransformerModel()

timeEnd = timeit.default_timer()
print("Time taken to train: ", timeEnd - timeBegin, "s\n")
print("\nWelcome to the Arsenal chat bot. You may ask me questions relating to Arsenal Football Club and I will do my best to answer them.")
print("I am also capable of simulating the evolution of any player based on your input. To perform a simulation please type 'Evolution'.\n")
print("You may also exit at any time by typing 'Quit'.\n\n")

while(flag == True):
    try:
        aimlFlag = False
                
        user_response = input ("Enter your message >> ")
        user_response = user_response.lower()

        trigger = 0 #Used to avoid having the chatbot respond with an answer from the question/answer base after certain parts

        while (user_response == ' '):
            user_response = input ("Enter your message >> ")
            user_response = user_response.lower()

        if (user_response == 'quit'):
            flag = False
            print("ArsenalBot: Goodbye!")
            break

        if (user_response == 'evolution'):
            print("Enter player name: ")
            playerName = input()
            while True:
                try:
                    potentialSeeked = float(input("\nEnter potential seeked (1-100): "))
                    if 1 <= potentialSeeked <= 100:
                        break
                    raise ValueError()
                except ValueError:
                    print("Input must be between 1-100!")

            while True:
                try:
                    playerSkill = float(input("\nEnter player current skill from bad to good(1-10): "))
                    if 1 <= playerSkill <= 10:
                        break
                    raise ValueError()
                except ValueError:
                    print("Input must be between 1-10!")    

            print("\n\n\n")
            potentialSimulator(playerName, potentialSeeked, playerSkill)
            trigger += 1
            

        if (user_response == 'can you tell me scores of past games?'):
            dateResponse = input ("Yes, please provide me a date: ") #01/05/2011
            output = predictMatch(dateResponse)
            print(output)
            print("\n")
            trigger += 1

        if (user_response == 'can you identify players?'):
            print ("Yes, I can tell you which Arsenal players are females and which are males.")
            path = input ("Please provide the directory with images that I can use to identify the players: ") #dataSet/testingData
            print ("\n")
            testingGenerator = createGenerator(path)
            predictImage(testingGenerator)
            trigger += 1 # +1 so the while loop skips the chatbot finding a suitable response to 'can you identify players?'
            
        if user_response in greetings:
            aimlFlag = True
        if user_response in farewells:
            aimlFlag = True
            flag = False
        if user_response in thanks:
            aimlFlag = True    
        if user_response in arsenal:
            aimlFlag = True
        if kernel.respond(user_response)[0] == '#':
            resp = kernel.respond(user_response)[1:].split('$')
            pattern = int(resp[0])
            if pattern == 0:
                print(resp[1])
                break

            
            if pattern == 1: #I WILL WATCH * PLAY ON *
                trigger += 1
                
                o = 'o' + str(objectCounter)
                objectCounter += 1
                val[ 'o' + o] = o
                if len(val[resp[1] ] ) ==1: 
                    if ' ' in val[resp[ 1] ]: 
                        val[resp[1] ].clear()
                if len(val[ "plays_on"] ) == 1: 
                    if (' ',) in val[ "plays_on"]:
                        val[ "plays_on"].clear()
                val[ "plays_on"].add((val[resp[1]], val[resp[ 2] ] ) ) #insert name and day
                print ("ArsenalBot: I'm sure " + resp[1].capitalize() + " will play well!")

             
            if pattern == 2: #IS * PLAYING ON *
                trigger += 1
                
                g = nltk.Assignment (val.domain)#map individual variables to entities in domain
                m = nltk.Model (val.domain, val)
                sent = resp[1] + ' plays_on ' + resp[2]
                results = nltk.evaluate_sents ([sent], grammarFile, m, g) [0] [0]
                if results [2] == True:
                    print ("Yes")
                else:
                    print("No")
                
            if pattern == 3: #WHICH PLAYER PLAYS ON *            
                trigger += 1

                g = nltk.Assignment (val.domain)
                m = nltk.Model (val.domain, val)
                e = nltk.Expression.fromstring("plays_on(x," + resp[1] + ")")
                sat = m.satisfiers (e, "x", g)
                if len(sat) == 0:
                    print ("None")
                else:
                    sol = val.values()
                    for so in sat:
                        for k, v in val.items():
                            if len(v) > 0:
                                vl = list(v)
                                if len(vl[0]) == 1:                                  
                                    for i in vl:
                                        if i[0] == so:
                                            print(k.capitalize())
                                            break
                                       

            
        if aimlFlag == True:
            print("ArsenalBot: ", kernel.respond(user_response))
            
        elif (trigger == 0):
            print("ArsenalBot: ", Response(user_response))
            
    except (KeyboardInterrupt, EOFError) as e:
        print("\nWhoops!")
        break
