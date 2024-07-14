Chatbot which takes in data from a user and returns if a loan should be given to him or not.
In case the loan was not given, the chatbot also explains why the loan was not given and on which aspects can the user improve to get the loan.
The chatbot first takes in input from the user, feed the input to our Neural Network model which provides the probabily of the user defaulting on the loan.
The Neural Network model also gives the different weightages it has given to different parameters provided by the user.
This is all the fed to the LLM by the chatbot, who then deciphers as to why the loan was not given by reading a document on which it has been trained on.
Making the LLM read a custom document tailor made for credit risk analyzing makes sure that the answers are financially and contextually correct along with being efficient.
