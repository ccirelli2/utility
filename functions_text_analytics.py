# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:32:44 2020

@author: chris.cirelli
"""

import string
import nltk
from nltk import corpus
from nltk.stem import *
from nltk.util import ngrams
stemmer = PorterStemmer()



def get_set_human_names():
    '''
    Purpose:  obtain a set of all human names
    Input:    none
    Output:   Set of of both female and male names
    '''
    # Create a lise of Male names, convert to lower case, split on '\n' as the text reads in a string as unicode
    Male_names = corpus.names.open('male.txt').read().lower().split('\n')
    # Create a lise of female names, convert to lower case, split on '\n' as the text reads in a string as unicode
    Female_names = corpus.names.open('female.txt').read().lower().split('\n')
    # Return to the user a set of the concatenation of both lists. 
    return set(Male_names + Female_names)

def clean_tokenize_text(Text_file, n_grams=False, num_grams=2):
    '''
    Input      = Text File
    Operations = Tokenize, lowercase, strip punctuation/stopwords/nonAlpha
    Return     = Object = Set; Set = cleaned, isalpha only tokens
    '''
    # Strip Lists
    Punct_list = set((punct for punct in string.punctuation))
    Stopwords = corpus.stopwords.words('english')
    Set_names = get_set_human_names()
    # Tokenize Text
    Text_tokenized = nltk.word_tokenize(Text_file)
    # Convert tokens to lowercase
    Text_lowercase = (token.lower() for token in Text_tokenized)
    # Strip Punctuation
    Text_tok_stripPunct = filter(lambda x: (x not in Punct_list), Text_lowercase)
    # Strip Stopwords
    Text_strip_stopWords = filter(lambda x: (x not in Stopwords), Text_tok_stripPunct)
    # Strip Non-Alpha
    Text_strip_nonAlpha = filter(lambda x: x.isalpha(), Text_strip_stopWords)
    # Strip 2 letter words
    Text_strip_2letter_words = filter(lambda x: len(x)>2, Text_strip_nonAlpha)
    # Strip names
    Text_strip_names_2 = filter(lambda x: x not in Set_names, Text_strip_2letter_words)
    # Take Stem of Each Token 
    Text_stem = [stemmer.stem(x) for x in Text_strip_names_2]
    
    if n_grams:
        n_grams = ngrams(Text_stem, num_grams)
        return n_grams
    else:
        return Text_stem


def get_token_freq(list_tokens):
    ''' Desc    Simple function to return the count by token from 
                a list of tokens. 
        Output  Dictionary whose key= token name and value count of that token. 
    '''
    token_freq_dict = dict()
    for token in list_tokens:
        token_freq_dict[token] = token_freq_dict.get(token, 0) + 1
    
    return token_freq_dict
