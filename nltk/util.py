#!/usr/bin/python
#coding: utf-8

import nltk

def tokenizeIt(sentence):
	tokenizer = nltk.tokenize.PunktWordTokenizer()
	return tokenizer.tokenize(sentence.lower())

def featureIt(tokens):
	return {}.fromkeys(tokens,True)

def featuresetIt(feature,labelList):
	return [ (feature,label) for label in labelList ]
