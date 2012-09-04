#!/usr/bin/python
#coding: utf-8

import nltk, csv
import util

class Tester:

	def __init__(self,trainedClassifier,csvReader):
		self.naive = trainedClassifier
		self.reader = csvReader

	def test(self,printTags=False,maxLines=0):
		lines = 0
		accepted = 0
		errors = 0
		for i in range(3):
			self.reader.next()
		for row in self.reader:
			title = row[6]
			body = row[7]
			question = (title, body)
			tags = []
			for i in range(8,13):
				if len(row[i])>0:
					tags += [row[i]]
			features = util.featureIt(util.tokenizeIt(title))
			features.update(util.featureIt(util.tokenizeIt(body)))

			maxProbTags = self.getMax(features)
			inputTags = [row[8],row[9],row[10],row[11],row[12]]
#			print "\t-------- TEST %d" % lines
#			print "Possible tags to the question #%s" % row[0]
#			print maxProbTags
#			print "Tags added"
#			print inputTags
			if printTags:
				print "\nTEST %d" % lines
				#print row[7]
				print "Tags found (%d): " % len(maxProbTags)
				print maxProbTags
				print "Tags marked:"
				print inputTags
				print '\n'

			for tag in inputTags:
				if not tag=="":
					if tag in maxProbTags:
						accepted+=1
					else:
						errors+=1
			print ("Testing line %d. Acc: %d, err: %d") % (lines,accepted, errors)
			lines+=1
			if lines>maxLines and maxLines>0:
				break
		print "Accepted: %d" % accepted
		print "Errors: %d" % errors		


	def getMax(self,features):
		prob = self.naive.prob_classify(features)
		tags = []
#		print "Found %d tag(s) that can fit" % len(prob.samples())
		for tag in prob.samples():
			for item in tags:
				pTag = prob.prob(tag)
				pItem = prob.prob(item)
				if pTag> (pItem+1.0e-15):
					tags.remove(item)
			if prob.prob(tag)>1.0e-15:
				tags.append(tag)
		dic = {}
		for tag in tags:
			dic.update({tag:prob.prob(tag)})
		return dic
