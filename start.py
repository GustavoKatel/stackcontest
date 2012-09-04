#!/usr/bin/python
#coding: utf-8

import nltk, csv, cPickle, os
import util,tester

MAX_LOADED_LINES=500

class stackClassifier:

	def train(self,filename):
		featureset=self.getFeatureset(filename)
		print "Training the classifier..."
		self.naive = nltk.NaiveBayesClassifier.train(featureset)

	def getTagAll(self,question):
		(title,body) = question	
		fe = util.featureIt(util.tokenizeIt(title)+util.tokenizeIt(body))
		return self.naive.prob_classify(fe)	.samples()

	def getTagMaxProb(self,question):
		(title,body) = question	
		fe = util.featureIt(util.tokenizeIt(title)+util.tokenizeIt(body))
		return self.naive.prob_classify(fe).max()

	def getProbI(self,question):
		(title,body) = question	
		fe = util.featureIt(util.tokenizeIt(title)+util.tokenizeIt(body))
		return self.naive.prob_classify(fe)

	def getNaiveObj(self):
		return self.naive

	def getCsvObj(self):
		return self.csvReader

	def saveFeaturesetFile(self,featureset,filename):
		fo = open(filename, "wb")
		version = 1.0
		cPickle.dump(version, fo, protocol = cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(featureset, fo, protocol = cPickle.HIGHEST_PROTOCOL)
		fo.close()

	def loadFeaturesetFile(self,filename):
		try:
			fo = open(filename, "rb")
		except IOError:
			# it's ok to not have the file
			print "didn't find file %s with data" % filename
			return
		try:
			version = cPickle.load(fo)
			fe = cPickle.load(fo)
		except:
			fo.close()
			print "Error loading the file: %s" % filename
			exit(1)
		fo.close()
		return fe

	def newFeatureset(self,filename):
		f = open(filename,"rb")
		self.csvReader = csv.reader(f)
		self.csvReader.next()
		featureset = []
		lines = 0
		for row in self.csvReader:
			title = row[6]
			body = row[7]
			tags = []
			for i in range(8,13):
				if len(row[i])>0:
					tags += [row[i]]
			featureset += util.featuresetIt( util.featureIt(util.tokenizeIt(title)), tags ) + util.featuresetIt( util.featureIt(util.tokenizeIt(body)), tags )
			lines+=1
			if MAX_LOADED_LINES>0 and lines>MAX_LOADED_LINES:
				print "Maximum exceeded!"
				break
			print "%d lines parsed." % lines
			#break
		self.saveFeaturesetFile(featureset,filename+".featureset")
		return featureset

	def getFeatureset(self,filename):
		if os.path.exists(filename+".featureset"):
			return self.loadFeaturesetFile(filename+".featureset")
		else:
			return self.newFeatureset(filename)

if __name__=="__main__":
	classifier = stackClassifier()
	classifier.train("/media/Arquivos/g4/NLP & ML/stackoverflow/train.csv")
	#

	#TESTER
	print "\nTESTER..."
	f = open("/media/Arquivos/g4/NLP & ML/stackoverflow/train.csv","rb")
	reader = csv.reader(f)
	reader.next()
	test = tester.Tester(classifier.getNaiveObj(),reader)#,classifier.getCsvObj())
	test.test(True,1)
