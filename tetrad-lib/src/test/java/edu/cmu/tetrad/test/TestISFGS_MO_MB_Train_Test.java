//package edu.cmu.tetrad.test;
//
//import java.io.File;
//import java.io.FileOutputStream;
//import java.io.PrintStream;
//import java.nio.file.Path;
//import java.nio.file.Paths;
//import java.util.ArrayList;
//import java.util.HashMap;
//import java.util.LinkedHashMap;
//import java.util.LinkedList;
//import java.util.List;
//import java.util.Map;
//import java.util.Map.Entry;
//import java.util.stream.Collectors;
//
//import edu.cmu.tetrad.bayes.*;
//import edu.cmu.tetrad.data.DataSet;
//import edu.cmu.tetrad.data.DataUtils;
//import edu.cmu.tetrad.data.DiscreteVariable;
//import edu.cmu.tetrad.data.IKnowledge;
//import edu.cmu.tetrad.data.Knowledge2;
//import edu.cmu.tetrad.graph.*;
//import edu.cmu.tetrad.search.*;
//import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetFileReader;
//import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetReader;
//import edu.cmu.tetrad.util.DataConvertUtils;
//import edu.cmu.tetrad.util.DelimiterUtils;
//
//public class TestISFGS_MO_MB_Train_Test {
//	public static void main(String[] args) {
//		
//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/PPT_data/";
//
//		String dataName = "ppt"; 
//		String pathToTrainData = pathToFolder + dataName + "_train_june28.csv";
//		String pathToTestData = pathToFolder + dataName + "_test_june28.csv";
//
//		String target1 = "Pain_Response";
//		String target2 = "PhysFunction_Response";
//		String target3 = "ImpChange_Response";
//		
//		// Read in the data
//		DataSet trainData = readData(pathToTrainData);
//		DataSet testData = readData(pathToTestData);	
//
//		// Create the knowledge
//		IKnowledge knowledge = null;//createKnowledge(trainData, target);
//
//		// learn the population model using all training data
//		double samplePrior = 1.0;
//		double structurePrior = 1.0;
//		Graph graphP = BNlearn_pop(trainData, knowledge, samplePrior, structurePrior);
//		System.out.println("Pop graph:" + graphP.getEdges());
//
//		double T_plus = 0.9;
//		double T_minus = 0.1;
//		
//		for (int p = 1; p <= 10; p++){
//
//			double k_add =  p/10.0; //Math.pow(10, -1.0*p);
//	
//			System.out.println("kappa = " + k_add);
//
//			double[] probs_is_1 = new double[testData.getNumRows()];
//			double[] probs_p_1 = new double[testData.getNumRows()];
//			int[] truth_1 = new int[testData.getNumRows()];
//			
//			double[] probs_is_2 = new double[testData.getNumRows()];
//			double[] probs_p_2 = new double[testData.getNumRows()];
//			int[] truth_2 = new int[testData.getNumRows()];
//			
//			double[] probs_is_3 = new double[testData.getNumRows()];
//			double[] probs_p_3 = new double[testData.getNumRows()];
//			int[] truth_3 = new int[testData.getNumRows()];
//			
//			Map <KeyMB, Double> stats= new HashMap<KeyMB, Double>();
//			PrintStream outForAUC;
//			try {
//				File dir = new File( pathToFolder + "/outputs/MB/" + dataName);
//				dir.mkdirs();
//				String outputFileName = dataName + "-AUROC-Kappa"+ k_add +".csv";
//				File fileAUC = new File(dir, outputFileName);
//				outForAUC = new PrintStream(new FileOutputStream(fileAUC));
//			} catch (Exception e) {
//				throw new RuntimeException(e);
//			}
//
//
//			outForAUC.println("y, population-FGES, instance-specific-FGES");//, DEGs");
//
//			// loop over test cases
//			for (int i = 0; i < testData.getNumRows(); i++){
//				
//				DataSet test = testData.subsetRows(new int[]{i});
//				
//				// learn the IS graph
//				Graph graphI = learnBNIS(trainData, test, k_add, graphP, knowledge, samplePrior);
//				
//				//get the prob from IS model
//				DagInPatternIterator iterator = new DagInPatternIterator(graphI);
//				Graph dagI = iterator.next();
//				dagI = GraphUtils.replaceNodes(dagI, trainData.getVariables());
//				Graph mbi_1 = GraphUtils.markovBlanketDag(dagI.getNode(target1), dagI);
//				Graph mbi_2 = GraphUtils.markovBlanketDag(dagI.getNode(target2), dagI);
//				Graph mbi_3 = GraphUtils.markovBlanketDag(dagI.getNode(target3), dagI);
//				
//				DagInPatternIterator iteratorP = new DagInPatternIterator(graphP);
//				Graph dagP = iteratorP.next();
//				dagP = GraphUtils.replaceNodes(dagP, trainData.getVariables());
//				Graph mbp_1 = GraphUtils.markovBlanketDag(dagP.getNode(target1), dagP);
//				Graph mbp_2 = GraphUtils.markovBlanketDag(dagP.getNode(target2), dagP);
//				Graph mbp_3 = GraphUtils.markovBlanketDag(dagP.getNode(target3), dagP);
//
//				
//				// compute probability distribution of the target variable
//				int[] targetIndices = new int[3];
//				targetIndices[0] = trainData.getColumn(trainData.getVariable(target1));
//				targetIndices[1] = trainData.getColumn(trainData.getVariable(target2)); 
//				targetIndices[2] = trainData.getColumn(trainData.getVariable(target3)); 
//
//				truth_1[i] = test.getInt(0, targetIndices[0]);
//				truth_2[i] = test.getInt(0, targetIndices[1]);
//				truth_3[i] = test.getInt(0, targetIndices[2]);
//
//
//				//get the prob from IS model
//				probs_is_1[i] = estimation(trainData, test, mbi_1, target1);
//				probs_is_2[i] = estimation(trainData, test, mbi_2, target2);
//				probs_is_3[i] = estimation(trainData, test, mbi_3, target3);
//
//				//get the prob from population model
//				probs_p_1[i] = estimation(trainData, test, mbp_1, target1);
//				probs_p_2[i] = estimation(trainData, test, mbp_2, target2);
//				probs_p_3[i] = estimation(trainData, test, mbp_3, target3);
//
//
////				//graph comparison
////				GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(mb_i, mb_p);
////				
////				int n_a = cmp.getEdgesAdded().size();
////				int n_d = cmp.getEdgesRemoved().size();
////				int n_r = cmp.getEdgesReorientedFrom().size();
////
////				KeyMB cur_key = new KeyMB(n_a, n_d, n_r);
////				if(stats.get(cur_key)!=null)
////					stats.put(cur_key, stats.get(cur_key) + 1.0);
////				else
////					stats.put(cur_key, 1.0);
//
//				outForAUC.println(test.getInt(0, targetIndices[0]) +", " + test.getInt(0, targetIndices[1]) + ", " + test.getInt(0, targetIndices[2]) 
//				+", " +  probs_p_1[i] + ", " +  probs_p_2[i] + ", " +  probs_p_3[i]+ ", " + 
//						probs_is_1[i] + ", " + probs_is_2[i] + ", " + probs_is_3[i]);			}
//			
//		//// target 1
//					System.out.println("------------ Pain_Response -------------");
//					double auroc_p_1 = AUC.measure(truth_1, probs_p_1);
//					double auroc_1 = AUC.measure(truth_1, probs_is_1);
//					System.out.println( "AUROC_Pop: "+ auroc_p_1);
//					System.out.println( "AUROC_IS : "+ auroc_1);
//
//					//// target 2
//					System.out.println("------------ PhysFunction_Response -------------");
//					double auroc_p_2 = AUC.measure(truth_2, probs_p_2);
//					double auroc_2 = AUC.measure(truth_2, probs_is_2);
//					System.out.println( "AUROC_Pop: "+ auroc_p_2);
//					System.out.println( "AUROC_IS : "+ auroc_2);
//
//					////  target 3
//					System.out.println("------------ ImpChange_Response -------------");
//					double auroc_p_3 = AUC.measure(truth_3, probs_p_3);
//					double auroc_3 = AUC.measure(truth_3, probs_is_3);
//					System.out.println( "AUROC_Pop: "+ auroc_p_3);
//					System.out.println( "AUROC_IS : "+ auroc_3);
//
//
////			for (KeyMB k : stats.keySet()){
////				System.out.println(k.print(k) + ":" + (stats.get(k)/testData.getNumRows())*100);
////			}
////			System.out.println("-----------------");
//			outForAUC.close();
//		}
//	}
//	
//	private static Graph learnBNIS(DataSet trainData, DataSet test, double kappa, Graph graphP, IKnowledge knowledge, double samplePrior){
//		// learn the instance-specific model
//		ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
//		scoreI.setSamplePrior(samplePrior);
//		scoreI.setKAddition(kappa);
//		scoreI.setKDeletion(kappa);
//		scoreI.setKReorientation(kappa);
//		ISFges fgesI = new ISFges(scoreI);
////		fgesI.setKnowledge(knowledge);
//		fgesI.setPopulationGraph(graphP);
//		fgesI.setInitialGraph(graphP);
//		Graph graphI = fgesI.search();
//		graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());
//		return graphI;
//	}
//
//	private static double estimation(DataSet trainData, DataSet test, Graph mb, String target){
//
//		double [] probs = classify(mb, trainData, test, (DiscreteVariable) test.getVariable(target));
//		return probs[1];
//	}
//
//	public static double[] classify(Graph mb, DataSet train, DataSet test, DiscreteVariable targetVariable) {
//
//		List<Node> mbNodes = mb.getNodes();
//
//		//The Markov blanket nodes will correspond to a subset of the variables
//		//in the training dataset.  Find the subset dataset.
//		DataSet trainDataSubset = train.subsetColumns(mbNodes);
//
//		//To create a Bayes net for the Markov blanket we need the DAG.
//		BayesPm bayesPm = new BayesPm(mb);
//
//		//To parameterize the Bayes net we need the number of values
//		//of each variable.
//		List varsTrain = trainDataSubset.getVariables();
//
//		for (int i1 = 0; i1 < varsTrain.size(); i1++) {
//			DiscreteVariable trainingVar = (DiscreteVariable) varsTrain.get(i1);
//			bayesPm.setCategories(mbNodes.get(i1), trainingVar.getCategories());
//		}
//
//		//Create an updater for the instantiated Bayes net.
//		DirichletBayesIm prior = DirichletBayesIm.symmetricDirichletIm(bayesPm, 1.0);
//		BayesIm bayesIm = DirichletEstimator.estimate(prior, trainDataSubset);
//
//		RowSummingExactUpdater updater = new RowSummingExactUpdater(bayesIm);
//
//		//The subset dataset of the dataset to be classified containing
//		//the variables in the Markov blanket.
//		List <Node> mb_Nodes_test = new ArrayList<Node> ();
//		for (Node n: mbNodes){
//			mb_Nodes_test.add(test.getVariable(n.getName()));
//		}
//		DataSet testSubset = test.subsetColumns(mb_Nodes_test);
//		
//		//Get the raw data from the dataset to be classified, the number
//		//of variables, and the number of cases.
//		double[] estimatedProbs = new double[targetVariable.getNumCategories()];
//
//		//The variables in the dataset.
//		List<Node> varsClassify = testSubset.getVariables();
//
//		//For each case in the dataset to be classified compute the estimated
//		//value of the target variable and increment the appropriate element
//		//of the crosstabulation array.
//
//		//Create an Evidence instance for the instantiated Bayes net
//		//which will allow that updating.
//		Proposition proposition = Proposition.tautology(bayesIm);
//
//		//Restrict all other variables to their observed values in
//		//this case.
//		int numMissing = 0;
//
//		for (int testIndex = 0; testIndex < varsClassify.size(); testIndex++) {
//			DiscreteVariable var = (DiscreteVariable) varsClassify.get(testIndex);
//
//			// If it's the target, ignore it.
//			if (var.equals(targetVariable)) {
//				continue;
//			}
//
//			int trainIndex = proposition.getNodeIndex(var.getName());
//
//			// If it's not in the train subset, ignore it.
//			if (trainIndex == -99) {
//				continue;
//			}
//
//			int testValue = testSubset.getInt(0, testIndex);
//
//			if (testValue == -99) {
//				numMissing++;
//			} else {
//				proposition.setCategory(trainIndex, testValue);
//			}
//		}
//
//		Evidence evidence = Evidence.tautology(bayesIm);
//		evidence.getProposition().restrictToProposition(proposition);
//		updater.setEvidence(evidence);
//
//		// for each possible value of target compute its probability in
//		// the updated Bayes net.  Select the value with the highest
//		// probability as the estimated getValue.
//		int targetIndex = proposition.getNodeIndex(targetVariable.getName());
//
//
//		for (int category = 0; category < targetVariable.getNumCategories(); category++) {
//			double marginal = updater.getMarginal(targetIndex, category);
//			estimatedProbs [category] = marginal;
//		}
//
//		return estimatedProbs;
//	}
//
//	private static Graph BNlearn_pop(DataSet trainDataOrig, IKnowledge knowledge, double samplePrior, double structurePrior) {
//		BDeuScore scoreP = new BDeuScore(trainDataOrig);
//		scoreP.setSamplePrior(samplePrior);
//		scoreP.setStructurePrior(structurePrior);
//		Fges fgesP = new Fges (scoreP);
////		fgesP.setKnowledge(knowledge);
//		fgesP.setSymmetricFirstStep(true);
//		Graph graphP = fgesP.search();
//		graphP = GraphUtils.replaceNodes(graphP, trainDataOrig.getVariables());
//		return graphP;
//	}
//	private static IKnowledge createKnowledge(DataSet trainDataOrig, String target) {
//		int numVars = trainDataOrig.getNumColumns();
//		IKnowledge knowledge = new Knowledge2();
//		int[] tiers = new int[2];
//		tiers[0] = 0;
//		tiers[1] = 1;
//		for (int i=0 ; i< numVars; i++) {
//			if (!trainDataOrig.getVariable(i).getName().equals(target)){
//				knowledge.addToTier(0, trainDataOrig.getVariable(i).getName());
//			}
//			else{
//				knowledge.addToTier(1, trainDataOrig.getVariable(i).getName());
//			}
//		}
//		knowledge.setTierForbiddenWithin(0, true);
//		return knowledge;
//	}
//	private static DataSet readData(String pathToData) {
//		Path trainDataFile = Paths.get(pathToData);
//		char delimiter = ',';
//		VerticalDiscreteTabularDatasetReader trainDataReader = new VerticalDiscreteTabularDatasetFileReader(trainDataFile, DelimiterUtils.toDelimiter(delimiter));
//		DataSet trainDataOrig = null;
//		try {
//			trainDataOrig = (DataSet) DataConvertUtils.toDataModel(trainDataReader.readInData());
//			System.out.println(trainDataOrig.getNumRows() +", " + trainDataOrig.getNumColumns());
//		} catch (Exception IOException) {
//			IOException.printStackTrace();
//		}
//		return trainDataOrig;
//	}
//	private static Map<String, Double> sortByValue(Map<String, Double> dEGdist, final boolean order)
//	{
//		List<Entry<String, Double>> list = new LinkedList<>(dEGdist.entrySet());
//
//		// Sorting the list based on values
//		list.sort((o1, o2) -> order ? o1.getValue().compareTo(o2.getValue()) == 0
//				? o1.getKey().compareTo(o2.getKey())
//						: o1.getValue().compareTo(o2.getValue()) : o2.getValue().compareTo(o1.getValue()) == 0
//						? o2.getKey().compareTo(o1.getKey())
//								: o2.getValue().compareTo(o1.getValue()));
//		return list.stream().collect(Collectors.toMap(Entry::getKey, Entry::getValue, (a, b) -> b, LinkedHashMap::new));
//
//	}
//}
