package com.stackcontest;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import com.csvreader.CsvReader;

import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.io.EncodingPrintWriter.out;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.ErasureUtils;

public class Main {

	private Main() {}
	
	protected static void serialize(LinearClassifier<String, Word> classifier)
	{
		try
		{
			File file = new File("stackcontest_data/classifier");
			FileOutputStream faos = new FileOutputStream(file);
			//ByteArrayOutputStream baos = new ByteArrayOutputStream();
		    ObjectOutputStream oos = new ObjectOutputStream(faos);
		    oos.writeObject(classifier);
		    oos.close();
		    faos.close();
		}catch (IOException e) {
			e.printStackTrace();
		}
	}
	protected static LinearClassifier<String, Word> deSerialize()
	{
		try
		{
			File file = new File("stackcontest_data/classifier");
			FileInputStream fin = new FileInputStream(file);
			ObjectInputStream obin = new ObjectInputStream(fin);
		    LinearClassifier<String,Word> lc = ErasureUtils.uncheckedCast(obin.readObject());
		    obin.close();
		    fin.close();
		    return lc;
		}catch (IOException e) {
			e.printStackTrace();
		}catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	protected static List<Word> tokenizeIt(String title, String body)
	{
		String titleLower = title.toLowerCase();
		String bodyLower = body.toLowerCase();
		List<Word> tokens = new ArrayList<Word>();
		
		String raw = titleLower+" "+bodyLower;
		StringReader reader = new StringReader(raw);

		PTBTokenizer<Word> tokensWords = PTBTokenizer.newPTBTokenizer(reader);
		
		while(tokensWords.hasNext())
		{
			Word w = tokensWords.next();
			tokens.add(w);
		}
		
		return tokens;
	}
	
	protected static Datum<String,Word> makeDatum(String title, String body, List<String> tags) {
	    List<Word> features = tokenizeIt(title, body);
	    return new BasicDatum<String, Word>(features,tags);
	  }
	
	public static void main(String args[]) throws IOException
	{
		
		LinearClassifierFactory<String,Word> factory = new LinearClassifierFactory<String,Word>();
		factory.useConjugateGradientAscent();
		factory.setVerbose(true);
		//
		LinearClassifier<String,Word> classifier = null;
		//
		if(new File("stackcontest_data/classifier").exists())
			classifier = LinearClassifier.readClassifier("stackcontest_data/classifier");
		else
		{
		    List<Datum<String, Word>> trainingData = new ArrayList<Datum<String,Word>>();
		    
		    int lineCount=0;
		    CsvReader reader=null;
		    try {
				reader = new CsvReader("stackcontest_data/train.csv", ',');
				reader.skipLine();
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		    while(reader.readRecord())
		    {
		    	String columns[] = reader.getValues();
		    	//System.out.println("columns "+columns.length);
		    	//
		    	ArrayList<String> tags = new ArrayList<String>();
		    	tags.add(columns[8]);
		    	if(!columns[9].isEmpty())
		    		tags.add(columns[9]);
		    	if(!columns[10].isEmpty())
		    		tags.add(columns[10]);
		    	if(!columns[11].isEmpty())
		    		tags.add(columns[11]);
		    	if(!columns[12].isEmpty())
		    		tags.add(columns[12]);
		    	//
		    	trainingData.add(makeDatum(columns[6], columns[7], tags));
		    	System.out.println("parsing line #"+lineCount);
		    	lineCount++;
		    	if(lineCount>1000)
		    		break;
		    }
		    
		    out.println("trainingData size: "+trainingData.size());
		    classifier = factory.trainClassifier(trainingData);
		    // Check out the learned weights
		    classifier.dump();
	    
		    //classifier.saveToFilename("stackcontest_data/classifier");
		    LinearClassifier.writeClassifier(classifier, "stackcontest_data/classifier");
		}
	    //
	    Datum<String, Word> test = makeDatum("Tools for porting J# code to C#", "Are there any conversion tools for porting Visual J# code to C#?", new ArrayList<String>());
	    out.println(classifier.classOf(test));
	    Counter c = classifier.probabilityOf(test);
	    for(int i=0;i<c.size();i++)
	    	out.println(c.keySet().toArray()[i]+": "+c.values().toArray()[i]);
	    	
	}
	
}
