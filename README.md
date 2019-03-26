# language-processing-technique
using CoreNLP, NLTK, and spacy to process text sample




to run the code

run the core NLP server with the command

# Run the server using all jars in the current directory (e.g., the CoreNLP home directory)
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

#then run the python file

python processText.py mysentences.text
