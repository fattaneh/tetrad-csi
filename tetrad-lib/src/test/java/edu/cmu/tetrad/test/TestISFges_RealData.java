package edu.cmu.tetrad.test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;
import edu.cmu.tetrad.bayes.*;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.data.Knowledge2;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
import edu.pitt.dbmi.data.reader.tabular.TabularDataReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDataReader;
import edu.cmu.tetrad.util.DataConvertUtils;
import edu.cmu.tetrad.util.DelimiterUtils;

public class TestISFges_RealData {
	private static final boolean True = false;
	private ConcurrentMap<Node, Integer> hashIndices ;
	public static void main(String[] args) {

		String workingDirectory = System.getProperty("user.dir");
		System.out.println(workingDirectory);

		Path trainDataFile = Paths.get("/Users/fattanehjabbari/CCD-Project/CS-BN/chronic_pancreatitis_fattane/train.csv");
		Path testDataFile = Paths.get("/Users/fattanehjabbari/CCD-Project/CS-BN/chronic_pancreatitis_fattane/test.csv");

		char delimiter = ',';

		TabularDataReader trainDataReader = new VerticalDiscreteTabularDataReader(trainDataFile.toFile(), DelimiterUtils.toDelimiter(delimiter));
		TabularDataReader testDataReader = new VerticalDiscreteTabularDataReader(testDataFile.toFile(), DelimiterUtils.toDelimiter(delimiter));
		DataSet trainData = null, testData = null;
		try {
			trainData = (DataSet) DataConvertUtils.toDataModel(trainDataReader.readInData());
			testData = (DataSet) DataConvertUtils.toDataModel(testDataReader.readInData());
			System.out.println(trainData.getNumRows() +", " + trainData.getNumColumns());
			System.out.println(testData.getNumRows() +", " + testData.getNumColumns());
		} catch (Exception IOException) {
			IOException.printStackTrace();
		}
		int numVars = trainData.getNumColumns();
		IKnowledge knowledge = new Knowledge2();
		int[] tiers = new int[2];
		tiers[0] = 0;
		tiers[1] = 1;
		for (int i=0 ; i< numVars; i++) {
			if (!trainData.getVariable(i).getName().equals("disease")){
				knowledge.addToTier(0, trainData.getVariable(i).getName());
			}
			else{
				knowledge.addToTier(1, trainData.getVariable(i).getName());
			}
		}
		knowledge.setTierForbiddenWithin(0, true);

		// learn the population model
		BDeuScore scoreP = new BDeuScore(trainData);
		scoreP.setSamplePrior(1.0);
		Fges fgesP = new Fges (scoreP);
		fgesP.setKnowledge(knowledge);
		Graph graphP = fgesP.search();
		graphP = GraphUtils.replaceNodes(graphP, trainData.getVariables());
		System.out.println(graphP);

		// estimate MAP parameters from the population model
		DagInPatternIterator iterator = new DagInPatternIterator(graphP);
		Graph dagP = iterator.next();
		dagP = GraphUtils.replaceNodes(dagP, trainData.getVariables());

		BayesPm pmP = new BayesPm(dagP);

		DirichletBayesIm priorP = DirichletBayesIm.symmetricDirichletIm(pmP, 1.0);
		BayesIm imP = DirichletEstimator.estimate(priorP, trainData);
		for (int p = 10; p <= 10; p++){
			
			double k_add =  p/10.0; //Math.pow(10, -1.0*p);
			double k_delete = 0.00000001;
			double k_reverse = k_add;
			System.out.println("kappa = " + k_add);

			double[] llr = new double[testData.getNumRows()];
			double average = 0.0;
			PrintStream out;
			PrintStream outForAUC;
			try {
				File dir = new File("/Users/fattanehjabbari/CCD-Project/CS-BN/simulation-newprior/");
				dir.mkdirs();
				String outputFileName = "Real-data-kadd" + k_add+"-kdel" + k_delete+"-krev" + k_reverse+".csv";
				File file = new File(dir, outputFileName);
				out = new PrintStream(new FileOutputStream(file));
				outputFileName = "Real-AUROC-data-kadd" + k_add+"-kdel" + k_delete+"-krev" + k_reverse+".csv";
				File fileAUC = new File(dir, outputFileName);
				outForAUC = new PrintStream(new FileOutputStream(fileAUC));
			} catch (Exception e) {
				throw new RuntimeException(e);
			}

			Map <Key, Double> stats= new HashMap<Key, Double>();
			outForAUC.println("disease, population, IS");
			for (int i = 0; i < testData.getNumRows(); i++){
				
				DataSet test = testData.subsetRows(new int[]{i});
				// learn the instance-specific model
				ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
				scoreI.setKAddition(k_add);
				scoreI.setKDeletion(k_delete);
				scoreI.setKReorientation(k_reverse);
				scoreI.setSamplePrior(1.0);
				ISFges fgesI = new ISFges(scoreI);
				fgesI.setKnowledge(knowledge);
				fgesI.setPopulationGraph(dagP);
				fgesI.setInitialGraph(dagP);
				Graph graphI = fgesI.search();
				graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());
				GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(graphI, graphP);
				int n_a = cmp.getEdgesAdded().size();
				int n_d = cmp.getEdgesRemoved().size();
				Key cur_key = new Key(n_a, n_d);
				if(stats.get(cur_key)!=null)
					stats.put(cur_key, stats.get(cur_key)+1.0);
				else
					stats.put(cur_key, 1.0);
	
				// learn a pop model from data + test
				DagInPatternIterator iteratorI = new DagInPatternIterator(graphI);
				Graph dagI = iteratorI.next();
				dagI = GraphUtils.replaceNodes(dagI, trainData.getVariables());
				BayesPm pmI = new BayesPm(dagI);
	
				DirichletBayesIm priorI = DirichletBayesIm.symmetricDirichletIm(pmI, 1.0);
				BayesIm imI = DirichletEstimator.estimate(priorI, trainData);
				fgesI.setPopulationGraph(dagP);
				llr[i] = fgesI.scoreDag(dagI) - fgesI.scoreDag(dagP);
				
	//			System.out.println("dagP: "+graphP.getEdges());
	//			System.out.println("dagI: "+graphI.getEdges());
	
				int diseaseIndex_p = imP.getNodeIndex(imP.getNode("disease"));
				int[] parents_p = imP.getParents(diseaseIndex_p);
				Arrays.sort(parents_p);
				int[] values_p = new int[parents_p.length];
			
				for (int no = 0; no < parents_p.length; no++){
					values_p [no] = test.getInt(0, parents_p[no]);
//					System.out.println(test.getVariable(parents_p[no]));
//					System.out.println(imP.getNode(parents_p[no]));
				}
				
				double prob_p = imP.getProbability(diseaseIndex_p, imP.getRowIndex(diseaseIndex_p, values_p), 1);
				
				int diseaseIndex_i = imI.getNodeIndex(imI.getNode("disease"));
				int[] parents_i = imI.getParents(diseaseIndex_i);
				Arrays.sort(parents_i);
				int[] values_i = new int[parents_i.length];
			
				for (int no = 0; no < parents_i.length; no++){
					values_i [no] = test.getInt(0, parents_i[no]);
				}
				double prob_i = imI.getProbability(diseaseIndex_i, imI.getRowIndex(diseaseIndex_i, values_i), 1);
	
				List<Node> parents_i_list = new ArrayList<Node>();
				for (int no = 0; no < parents_i.length; no++){
					parents_i_list.add(imI.getNode(parents_i[no]));
				}
				
	//			System.out.println(dagI.getNode("disease"));
	//			System.out.println(diseaseIndex_i);
	//			System.out.println(imI.getRowIndex(diseaseIndex_i, values_i));
//				System.out.println(parents_i_list);
	//			System.out.println(Arrays.toString(parents_i));
	//			System.out.println(imP.toString(diseaseIndex_i));
//				System.out.println("parent values: "+ Arrays.toString(values_p)); 
//				System.out.println("P_p(disease = 1) = " + prob_p);
	
	//			System.out.println(imI.toString(diseaseIndex_i));
//				System.out.println("parent values: "+ Arrays.toString(values_i)); 
//				System.out.println("target value:  " + test.getInt(0, diseaseIndex_p));
//				System.out.println("P_p(disease = 1) = " + prob_p);
//				System.out.println("P_i(disease = 1) = " + prob_i);
				outForAUC.println(test.getInt(0,diseaseIndex_i) +", " + prob_p + ", "+ prob_i);
//				System.out.println("-----------------");
				average += llr[i];
			}
			for (Key k : stats.keySet()){
				System.out.println(k.print(k) + ":" + (stats.get(k)/testData.getNumRows())*100);
			}
			out.println(Arrays.toString(llr));
			out.close();
			outForAUC.close();
			System.out.println(average/testData.getNumRows());
		}
		System.out.println("-----------------");

	}
}
class Key {

	public final int n_a;
	public final int n_d;

	public Key(final int n_a, final int n_d) {
		this.n_a = n_a;
		this.n_d = n_d;
	}
	@Override
	public boolean equals (final Object O) {
		if (!(O instanceof Key)) return false;
		if (((Key) O).n_a != n_a) return false;
		if (((Key) O).n_d != n_d) return false;
		return true;
	}
	 @Override
	 public int hashCode() {
		 return this.n_a ^ this.n_d;
	 }
	 public String print(Key key){
		return "("+key.n_a +", "+ key.n_d + ")";
	 }

}
