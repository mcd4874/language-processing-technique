"""
author: william duong
This file will apply coreNLP to perform word_tokenize, POS tag, name entityt recognition, annotation, lemmatize and dependency parse

"""
from stanfordcorenlp import StanfordCoreNLP
import json
from collections import defaultdict
import pandas as pd
import sys

class StanfordNLP:
    """
    this class will create funciton to process text using coreNLP

    """
    def __init__(self, host='http://localhost', port=9000):
        """
        will initialize the coreNLP server through localhost with port 9000
        :param host:
        :param port:
        """
        self.nlp = StanfordCoreNLP(host, port=port,
        timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
        'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
        'pipelineLanguage': 'en',
        'outputFormat': 'json'
                }
    def word_tokenize(self, sentence):
        """tokenize using standford core NLP"""
        return self.nlp.word_tokenize(sentence)
    def pos(self, sentence):
        """pos tag using standford core NLP"""
        return self.nlp.pos_tag(sentence)
    def ner(self, sentence):
        """name entity recognition  using standford core NLP"""
        return self.nlp.ner(sentence)
    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)
    def annotate(self, sentence):
        """annotate using standford core NLP"""
        return json.loads(self.nlp.annotate(sentence, properties=self.props))
    def lemmatize(self,sentence):
        """lemmatize using standford core NLP"""
        parsed_dict =self.annotate(sentence)
        lemma_list = [v for b in parsed_dict['sentences'] for d in b['tokens'] for k, v in d.items() if k == 'lemma']
        " ".join(lemma_list)
        return lemma_list

    def sentence_split(self,sentence):
        """Split sentence using Stanford NLP"""
        annotated = self.annotate(sentence)
        sentence_split = list()
        for sentence in annotated['sentences']:
            s = [t['word'] for t in sentence['tokens']]
            sentence_split.append(" ".join(s))
        return sentence_split

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
                            }
        return tokens


def process_text_with_core_nlp(text1,tableName):
    """
    this function will process input sentences using coreNLp
    :param text1: input sentence
    :param tableName: the pandas table name to generate result
    :return:
    """
    sNLP = StanfordNLP()
    token1 = sNLP.word_tokenize(text1)
    pos1 = sNLP.pos(text1)
    ner1 = sNLP.ner(text1)
    lemmatize1 = sNLP.lemmatize(text1)
    dependency_parse = sNLP.dependency_parse(text1)
    table_list = list()
    for i in range(len(token1)):
        current = [token1[i], lemmatize1[i],pos1[i][1], ner1[i][1], dependency_parse[i]]
        table_list.append(current)
    table1 = pd.DataFrame(table_list,columns=["token","lemmatize","POS","NameEntityRecog","dependency_parse"])
    print("coreNLP information : ")
    print (table1)
    table1.to_csv(tableName)
    print("sentence split : ", sNLP.sentence_split(text1))

def process_text_with_spacy(text1,tableName):
    """
    this function will process input sentences using spacy
    :param text1: input sentence
    :param tableName: the pandas table name to generate result
    :return:
    """
    import spacy
    sp = spacy.load('en_core_web_sm')
    doc = sp(text1)
    table_list2 = list()
    for X in doc:
        depend = "{0} <- {1} <- {2}".format(
            X.text, X.dep_, X.head.text)
        current = [X, X.lemma_, X.pos_, X.ent_type_, depend]
        table_list2.append(current)
    table2 = pd.DataFrame(table_list2,columns=["token","lemmatize","POS","NameEntityRecog","dependency_parse"])
    print("spacy information : ")
    print(table2)
    table2.to_csv(tableName)
    result = list()
    for sent in doc.sents:
        result.append(sent.text)
    print("spacy sentences split: ",result)





def main():
    textFile = sys.argv[1]
    file = open(textFile,'r')
    # romeo and juliet
    text1 = file.readline().rstrip()
    # julius casear
    text2 = file.readline().rstrip()
    process_text_with_core_nlp(text1,"text1_coreNLP_table.csv")
    process_text_with_spacy(text1,"text1_spacy_table.csv")

    process_text_with_core_nlp(text2,"text2_coreNLP_table.csv")
    process_text_with_spacy(text2,"text2_spacy_table.csv")
main()

