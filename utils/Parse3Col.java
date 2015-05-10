/*
javac ParseThreeCol.java -cp ./stanford-parser.jar:./stanford-parser-sources.jar:/u/nlp/data/StanfordCoreNLPModels/stanford-lexparser-models-current.jar:stanford-corenlp-3.3.1.jar:stanford-corenlp-3.3.1-sources.jar:/u/nlp/data/StanfordCoreNLPModels/stanford-corenlp-models-current.jar:guava-18.0.jar
java -cp ./stanford-parser.jar:./stanford-parser-sources.jar:/u/nlp/data/StanfordCoreNLPModels/stanford-lexparser-models-current.jar:stanford-corenlp-3.3.1.jar:stanford-corenlp-3.3.1-sources.jar:/u/nlp/data/StanfordCoreNLPModels/stanford-corenlp-models-current.jar:guava-18.0.jar:. ParseThreeCol unparsed_entailment_pairs.tsv > parsed_entailment_pairs.tsv 
*/

import java.util.Collection;
import java.util.List;
import java.io.StringReader;
import java.io.FileReader;
import java.io.BufferedReader;

import com.google.common.cache.*;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.parser.lexparser.*;
import edu.stanford.nlp.sentiment.CollapseUnaryTransformer;

class ParseThreeCol {
	
	/**
	 * The main method demonstrates the easiest way to load a parser.
	 * Simply call loadModel and specify the path of a serialized grammar
	 * model, which can be a file, a resource on the classpath, or even a URL.
	 * For example, this demonstrates loading from the models jar file, which
	 * you therefore need to include in the classpath for ParserDemo to work.
	 */
	public static void main(String[] args) {
		final LexicalizedParser lp = LexicalizedParser.loadModel("./englishPCFG.caseless.np_biased.ser.gz");
		final TokenizerFactory<CoreLabel> tokenizerFactory =
		    PTBTokenizer.factory(new CoreLabelTokenFactory(), "");
		final CollapseUnaryTransformer transformer = new CollapseUnaryTransformer();
		final TreeBinarizer binarizer = new TreeBinarizer(lp.getTLPParams().headFinder(), lp.treebankLanguagePack(), 
													false, false, 0, false, false, 0.0, false, true, true);
					
		LoadingCache<String, String> parses = CacheBuilder.newBuilder()
		       .maximumSize(120000)
		       .build(
		           new CacheLoader<String, Tree>() {
		             public String load(String sentence) throws Exception {
		             					Tokenizer<CoreLabel> tok =
										  tokenizerFactory.getTokenizer(new StringReader(sentence));
										List<CoreLabel> rawWords2 = tok.tokenize();
										Tree parse = lp.apply(rawWords2);
										return parse;
		             }
		           });

		try {
			BufferedReader br = new BufferedReader(new FileReader(args[0]));  
			String line = null;  
			String[] columnDetail = new String[3];
			String[] binarized = new String[2];
			String[] parsed = new String[2];

			int num_done = 0;
			while ((line = br.readLine()) != null) {
				try {
					columnDetail = line.split("\t");
					for (int i = 1; i < 3; i++) {
						Tree parse = parses.get(columnDetail[i]);	
						Tree bin = binarizer.transformTree(parse);		
						Tree collapsed = transformer.transformTree(bin);
						return unlabeledPrint(collapsed)
						parsed[i - 1] = parse.taggedYield();	
						binarized[i - 1] = unlabeledPrint(collapsed);	
					}
					System.binarized.println(columnDetail[0] + "\t" + binarized[0] + "\t" + binarized[1] + "\t" + columnDetail[1] + "\t" + columnDetail[2]);
					num_done++;
					if (num_done % 1000 == 0) {
						System.err.println("Finished " + num_done + ".");
					}
				}  catch (Exception e) {
					System.err.println(e);
					System.err.println("\tFor line: " + line);
				}
			} 
		} catch (Exception e) {
			System.err.println(e);
		}
	}
	
	static String unlabeledPrint(Tree tree) {
		if (tree.isLeaf()) {
			return tree.nodeString();
		} else if (tree.isPreTerminal()) {
			for (Tree child : tree.children()) {
				return unlabeledPrint(child);
			}
			return "---";
		} else {
			String rv = "(";
			for (Tree child : tree.children()) {
				rv = rv + " " + unlabeledPrint(child); 
			}
			return rv + " )";
		}
	}
	
	
	private ParseThreeCol() {} // static methods only
	
}
