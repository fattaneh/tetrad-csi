package edu.cmu.tetrad.test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import edu.cmu.tetrad.bayes.*;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.data.Knowledge2;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
import edu.cmu.tetrad.test.Key;
import edu.pitt.dbmi.data.reader.tabular.TabularDataReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetFileReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetReader;
import edu.cmu.tetrad.util.DataConvertUtils;
import edu.cmu.tetrad.util.DelimiterUtils;

public class TestISFGS_LOOCV {
	public static void main(String[] args) {
		
		
		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/Shyam-data/";
		String dataName = "genims_mortality_4i";
		String pathToData = pathToFolder + "GenIMS/" + dataName + ".csv";
		String target = "day90_status";
		
//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/Shyam-data/";
//		String dataName = "genims_sepsis_5i";
//		String pathToData = pathToFolder + "GenIMS/" + dataName + ".csv";
//		String target = "everss";
		
//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/Shyam-data/";
//		String dataName = "port_all";
//		String pathToData = pathToFolder + "PORT/" + dataName + ".csv";
//		String target = "217.DIREOUT";
	
//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/Shyam-data/";
//		String dataName = "heart_death_di";
//		String pathToData = pathToFolder + "Heart Failure/" + dataName + ".csv";
//		String target = "death";
		
		// Read in the data
		DataSet trainDataOrig = readData(pathToData);
		
		// Create the knowledge
		IKnowledge knowledge = createKnowledge(trainDataOrig, target);

		// learn the population model using all training data
		double samplePrior = 1.0;
		double structurePrior = 1.0;
		Graph graphP = BNlearn_pop(trainDataOrig, knowledge, samplePrior, structurePrior);
		System.out.println("Pop graph:" + graphP.getEdges());

		double T_plus = 0.9;
		double T_minus = 0.1;
		
		for (int p = 1; p <= 10; p++){

			double k_add =  p/10.0; //Math.pow(10, -1.0*p);
	
			System.out.println("kappa = " + k_add);

			double[] probs_is = new double[trainDataOrig.getNumRows()];
			double[] probs_p = new double[trainDataOrig.getNumRows()];
			int[] truth = new int[trainDataOrig.getNumRows()];
			Map <Key, Double> stats= new HashMap<Key, Double>();
//			PrintStream out;
			PrintStream outForAUC;
			try {
				File dir = new File( pathToFolder + "/outputs/" + dataName);
				dir.mkdirs();
				String outputFileName = dataName + "-AUROC-Kappa"+ k_add +".csv";
				File fileAUC = new File(dir, outputFileName);
				outForAUC = new PrintStream(new FileOutputStream(fileAUC));
			} catch (Exception e) {
				throw new RuntimeException(e);
			}


			outForAUC.println("y, population-FGES, instance-specific-FGES");//, DEGs");

			//LOOCV loop
			for (int i = 0; i < trainDataOrig.getNumRows(); i++){
				
				DataSet trainData = trainDataOrig.copy();
				DataSet test = trainDataOrig.subsetRows(new int[]{i});
				trainData.removeRows(new int[]{i});
				
				// learn the IS graph
				Graph graphI = learnBNIS(trainData, test, k_add, graphP, knowledge, samplePrior);

				// compute probability distribution of the target variable
				int targetIndex = trainData.getColumn(trainData.getVariable(target)); //imP.getNodeIndex(imP.getNode(target));
				truth[i] = test.getInt(0, targetIndex);
				
				//get the prob from IS model
				probs_is[i]= estimation(trainData, test, graphI, target);

				//get the prob from population model
				double prob_p = estimation(trainData, test, graphP, target);
				probs_p[i] = prob_p; 

				//graph comparison
				GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(graphI, graphP);
				int n_a = cmp.getEdgesAdded().size();
				int n_d = cmp.getEdgesRemoved().size();

//				if (n_d==1 && n_a==0){
//					System.out.println("case_i: " + i);
//					System.out.println("graph_i: " + graphI.getEdges());
//				
//					BDeuScore score1 = new BDeuScore(trainData);
//					ISBDeuScore score2 = new ISBDeuScore(trainData, test);
//					//				
//					int [] parents_old = new int[1];
//					parents_old[0] = trainData.getColumn(trainData.getVariable("F17"));
//
//					int [] parents_new = new int[2];
//					parents_new[0] = trainData.getColumn(trainData.getVariable("F13"));
//					parents_new[1] = trainData.getColumn(trainData.getVariable("F17"));
//
//					System.out.println("i score with 1 parents" + score2.localScore(0, parents_old, parents_new , new int[0]));
//					System.out.println("i score with 2 parents" + score2.localScore(0, parents_new,  parents_new , new int[0]));
//					System.out.println("i score pop with 2 parents" + score1.localScore(0, parents_new));
//
//					System.out.println("-------------");
//				}

				Key cur_key = new Key(n_a, n_d);
				if(stats.get(cur_key)!=null)
					stats.put(cur_key, stats.get(cur_key) + 1.0);
				else
					stats.put(cur_key, 1.0);

				outForAUC.println(test.getInt(0, targetIndex) +", " + probs_p[i] + ", "+ probs_is[i]);//+ ", " + parents_i_list.toString());
			}
			double auroc_p = AUC.measure(truth, probs_p);
			double auroc = AUC.measure(truth, probs_is);
			
			double fcr_p = FCR.measure(truth, probs_p, T_plus, T_minus);
			double fcr = FCR.measure(truth, probs_is, T_plus, T_minus);

			System.out.println( "AUROC_P: "+ auroc_p);
			System.out.println( "AUROC: "+ auroc);
			System.out.println( "FCR_P: "+ fcr_p);
			System.out.println( "FCR: "+ fcr);

			for (Key k : stats.keySet()){
				System.out.println(k.print(k) + ":" + (stats.get(k)/trainDataOrig.getNumRows())*100);
			}
			System.out.println("-----------------");
			outForAUC.close();
		}
	}
	
	private static Graph learnBNIS(DataSet trainData, DataSet test, double kappa, Graph graphP, IKnowledge knowledge, double samplePrior){
		// learn the instance-specific model
		ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
		scoreI.setSamplePrior(samplePrior);
		scoreI.setKAddition(kappa);
		scoreI.setKDeletion(kappa);
		scoreI.setKReorientation(kappa);
		ISFges fgesI = new ISFges(scoreI);
		fgesI.setKnowledge(knowledge);
		fgesI.setPopulationGraph(graphP);
		fgesI.setInitialGraph(graphP);
		Graph graphI = fgesI.search();
		graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());
		return graphI;
	}

	private static double estimation(DataSet trainData, DataSet test, Graph graph, String target){

		DagInPatternIterator iterator = new DagInPatternIterator(graph);
		Graph dag = iterator.next();
		dag = GraphUtils.replaceNodes(dag, trainData.getVariables());
		BayesPm pm = new BayesPm(dag);

		DirichletBayesIm prior = DirichletBayesIm.symmetricDirichletIm(pm, 1.0);
		BayesIm im = DirichletEstimator.estimate(prior, trainData);

//		System.out.println("mbn: " + dag.getParents(dag.getNode(target)));	
		int targetIndex = im.getNodeIndex(im.getNode(target));
		int[] parents = im.getParents(targetIndex);
		Arrays.sort(parents);
		int[] values = new int[parents.length];
		for (int no = 0; no < parents.length; no++){
			values [no] = test.getInt(0, parents[no]);
		}
		double prob = im.getProbability(targetIndex, im.getRowIndex(targetIndex, values), 1);
		return prob;
	}
//	private static double kappaTuning(DataSet trainData, IKnowledge knowledge, Graph dagP, String target){
//		//inner LOOCV
//		int kvalues = 9;
//		double[] kappa_tune = new double[kvalues+1];
//		double[] auroc_tune = new double[kvalues+1];
//		for (int p = 0; p <= kvalues; p++){	
//			double k_add =  (p+1)/10.0; 
//			double k_delete = k_add;
//			double k_reverse = k_add;
//			kappa_tune[p]= k_add;
//			System.out.println("kappa: " + k_add);
//
//			double[] probs_is_tune = new double[trainData.getNumRows()];
//			int[] truth_tune = new int[trainData.getNumRows()];
//			for (int j = 0; j < trainData.getNumRows(); j++){
//				DataSet trainData66 = trainData.copy();
//				DataSet test66 = trainData.subsetRows(new int[]{j});
//				trainData66.removeRows(new int[]{j});
//
//				// learn the instance-specific model
//				//				System.out.println("trainData66: "+trainData66.getNumRows());
//				ISBDeuScore scoreI = new ISBDeuScore(trainData66, test66);
//				scoreI.setKAddition(k_add);
//				scoreI.setKDeletion(k_delete);
//				scoreI.setKReorientation(k_reverse);
//				ISFges fgesI = new ISFges(scoreI);
//				fgesI.setKnowledge(knowledge);
//				fgesI.setPopulationGraph(dagP);
//				fgesI.setInitialGraph(dagP);
//				Graph graphI = fgesI.search();
//				graphI = GraphUtils.replaceNodes(graphI, trainData66.getVariables());
//
//				// learn a pop model from data + test
//				DagInPatternIterator iteratorI = new DagInPatternIterator(graphI);
//				Graph dagI = iteratorI.next();
//				dagI = GraphUtils.replaceNodes(dagI, trainData66.getVariables());
//				BayesPm pmI = new BayesPm(dagI);
//
//				DirichletBayesIm priorI = DirichletBayesIm.symmetricDirichletIm(pmI, 1.0);
//				BayesIm imI = DirichletEstimator.estimate(priorI, trainData66);
//				fgesI.setPopulationGraph(dagP);
//
//				int diseaseIndex_i = imI.getNodeIndex(imI.getNode(target));
//				truth_tune [j] = test66.getInt(0,diseaseIndex_i);
//
//				double prob_i = estimation_pop(imI, test66, diseaseIndex_i);
//				probs_is_tune[j] = prob_i; 
//			}
//
//			auroc_tune[p] = FCR.measure(truth_tune, probs_is_tune, 0.85, 0.15);
//			//			System.out.println( "AUROC: "+ auroc_tune[p]);
//		}
//
//		QuickSort.sort(auroc_tune, kappa_tune);
//
//
//		double best_kappa = kappa_tune[kvalues];
//		double best_auroc = auroc_tune[kvalues];
//		if (best_auroc == 0.0){
//			best_kappa = 0.3;
//		}
//		System.out.println( "BEST fcr: "+ best_auroc + ", BEST kappa: " + best_kappa);
//		return best_kappa;
//	}

	private static Graph BNlearn_pop(DataSet trainDataOrig, IKnowledge knowledge, double samplePrior, double structurePrior) {
		BDeuScore scoreP = new BDeuScore(trainDataOrig);
		scoreP.setSamplePrior(samplePrior);
		scoreP.setStructurePrior(structurePrior);
		Fges fgesP = new Fges (scoreP);
		fgesP.setKnowledge(knowledge);
		fgesP.setSymmetricFirstStep(true);
		Graph graphP = fgesP.search();
		graphP = GraphUtils.replaceNodes(graphP, trainDataOrig.getVariables());
		return graphP;
	}
	private static IKnowledge createKnowledge(DataSet trainDataOrig, String target) {
		int numVars = trainDataOrig.getNumColumns();
		IKnowledge knowledge = new Knowledge2();
		int[] tiers = new int[2];
		tiers[0] = 0;
		tiers[1] = 1;
		for (int i=0 ; i< numVars; i++) {
			if (!trainDataOrig.getVariable(i).getName().equals(target)){
				knowledge.addToTier(0, trainDataOrig.getVariable(i).getName());
			}
			else{
				knowledge.addToTier(1, trainDataOrig.getVariable(i).getName());
			}
		}
		knowledge.setTierForbiddenWithin(0, true);
		return knowledge;
	}
	private static DataSet readData(String pathToData) {
		Path trainDataFile = Paths.get(pathToData);
		char delimiter = ',';
		VerticalDiscreteTabularDatasetReader trainDataReader = new VerticalDiscreteTabularDatasetFileReader(trainDataFile, DelimiterUtils.toDelimiter(delimiter));
		DataSet trainDataOrig = null;
		try {
			trainDataOrig = (DataSet) DataConvertUtils.toDataModel(trainDataReader.readInData());
			System.out.println(trainDataOrig.getNumRows() +", " + trainDataOrig.getNumColumns());
		} catch (Exception IOException) {
			IOException.printStackTrace();
		}
		return trainDataOrig;
	}
	private static Map<String, Double> sortByValue(Map<String, Double> dEGdist, final boolean order)
	{
		List<Entry<String, Double>> list = new LinkedList<>(dEGdist.entrySet());

		// Sorting the list based on values
		list.sort((o1, o2) -> order ? o1.getValue().compareTo(o2.getValue()) == 0
				? o1.getKey().compareTo(o2.getKey())
						: o1.getValue().compareTo(o2.getValue()) : o2.getValue().compareTo(o1.getValue()) == 0
						? o2.getKey().compareTo(o1.getKey())
								: o2.getValue().compareTo(o1.getValue()));
		return list.stream().collect(Collectors.toMap(Entry::getKey, Entry::getValue, (a, b) -> b, LinkedHashMap::new));

	}
}