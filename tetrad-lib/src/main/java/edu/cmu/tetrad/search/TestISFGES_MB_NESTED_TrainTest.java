package edu.cmu.tetrad.search;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import edu.cmu.tetrad.bayes.*;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.graph.*;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetFileReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetReader;
import edu.cmu.tetrad.util.DataConvertUtils;
import edu.cmu.tetrad.util.DelimiterUtils;

public class TestISFGES_MB_NESTED_TrainTest {
	public static void main(String[] args) {

		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/dissertation/Shyam-data/";
		String pathToKfold =  pathToFolder + "GenIMS/";
		String dataName = "genims_mortality_4i";		
		String pathToTrainData = pathToFolder + "GenIMS/" + dataName + "_train.csv";
		String pathToTestData = pathToFolder + "GenIMS/" + dataName + "_test.csv";
		String target = "day90_status";
		
//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/Shyam-data/";
//		String dataName = "genims_sepsis_5i";
//		String pathToData = pathToFolder + "GenIMS/" + dataName + ".csv";
//		String target = "everss";

//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/dissertation/Shyam-data/";
//		String dataName = "port";
//		String pathToTrainData = pathToFolder + "PORT/" + dataName + "_train.csv";
//		String pathToTestData = pathToFolder + "PORT/" + dataName + "_test.csv";
//		String pathToKfold =  pathToFolder + "PORT/";
//		String target = "217.DIREOUT";



		// Read in the data
		DataSet trainData = readData(pathToTrainData);
		DataSet testData = readData(pathToTestData);
		System.out.println("train: " + trainData.getNumRows() +", " + trainData.getNumColumns()+ ", test: " + testData.getNumRows() +", " + testData.getNumColumns());

		// learn the population model using all training data
		double samplePrior = 1.0;
		double structurePrior = 1.0;
		Graph graphP = BNlearn_pop(trainData, samplePrior, structurePrior);
		System.out.println("Pop graph:" + graphP.getEdges());
		PrintStream logFile;
		try {
			File dir = new File( pathToFolder + "/IGES-Tunned/MB/" + dataName + "/PESS" + samplePrior);
			dir.mkdirs();
			String outputFileName = dataName + "PESS" + samplePrior +"_log.txt";
			File fileAUC = new File(dir, outputFileName);
			logFile = new PrintStream(new FileOutputStream(fileAUC));

		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		
		System.out.println("PESS = " + samplePrior);
		logFile.println("PESS = " + samplePrior);
		
		logFile.println("train: " +trainData.getNumRows() +", " + trainData.getNumColumns());
		logFile.println("test: " +testData.getNumRows() +", " + testData.getNumColumns());
		
		logFile.println("Pop graph:" + graphP.getEdges());

		//outer LOOCV
		double[] probs_is = new double[testData.getNumRows()];
		double[] probs_p = new double[testData.getNumRows()];
		int[] truth = new int[testData.getNumRows()];

		PrintStream outForAUC, out;
		try {
			File dir = new File( pathToFolder + "/IGES-Tunned/MB/" + dataName + "/PESS" + samplePrior);
			dir.mkdirs();
			String outputFileName = dataName + "-AUROC-PESS" + samplePrior +".csv";
			File fileAUC = new File(dir, outputFileName);
			outForAUC = new PrintStream(new FileOutputStream(fileAUC));

			outputFileName = dataName + "FeatureDist-PESS" + samplePrior +".csv";
			File filePredisctors = new File(dir, outputFileName);
			out = new PrintStream(new FileOutputStream(filePredisctors));

		} catch (Exception e) {
			throw new RuntimeException(e);
		}

		Map <KeyMB, Double> stats= new HashMap<KeyMB, Double>();
		Map <String, Double> fdist= new HashMap<String, Double>();
		for (int i = 0; i < testData.getNumColumns(); i++){
			fdist.put(testData.getVariable(i).getName(), 0.0);
		}
		System.out.println(fdist);

		out.println("features, fraction of occurance in cases");
		outForAUC.println("y, population-FGES, instance-specific-FGES, kappa");//, DEGs");

		// Outer LOOCV
		for (int i = 0; i < testData.getNumRows(); i++){

			System.out.println("i = " + i);
			DataSet test = testData.subsetRows(new int[]{i});

			int targetIndex = testData.getColumn(testData.getVariable(target)); 
			truth[i] = test.getInt(0, targetIndex);

			// estimate MAP parameters from the population model
			DagInPatternIterator iteratorP = new DagInPatternIterator(graphP);
			Graph dagP = iteratorP.next();
			dagP = GraphUtils.replaceNodes(dagP, trainData.getVariables());
			Graph dagPt = GraphUtils.replaceNodes(dagP, test.getVariables());

			Graph mb_p = GraphUtils.markovBlanketDag(dagP.getNode(target), dagP);
			Graph mb_pt = GraphUtils.markovBlanketDag(dagPt.getNode(target), dagPt);
			probs_p[i] = estimation(trainData, test, (Dag) mb_p, (Dag) mb_pt, target);

			// tune kappa for is model
			double best_kappa = kappaTuning(pathToKfold, dataName, samplePrior, dagP, target, targetIndex);
			Graph graphI = learnBNIS(trainData, test, best_kappa, graphP, samplePrior);

			//get the prob from IS model
			DagInPatternIterator iterator = new DagInPatternIterator(graphI);
			Graph dagI = iterator.next();
			dagI = GraphUtils.replaceNodes(dagI, trainData.getVariables());
			Graph dagIt = GraphUtils.replaceNodes(dagI, test.getVariables());

			Graph mb_i = GraphUtils.markovBlanketDag(dagI.getNode(target), dagI);
			Graph mb_it = GraphUtils.markovBlanketDag(dagIt.getNode(target), dagIt);

			probs_is[i]= estimation(trainData, test, (Dag) mb_i, (Dag) mb_it, target);

			List<Node> mb_nodes = mb_i.getNodes();
			mb_nodes.remove(mb_i.getNode(target));
			for (Node no: mb_nodes){
//				System.out.println(no);
				fdist.put(no.getName(), fdist.get(no.getName()) + 1.0);
			}

			//graph comparison
			GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(graphI, graphP);
			int n_a = cmp.getEdgesAdded().size();
			int n_d = cmp.getEdgesRemoved().size();
			int n_r = cmp.getEdgesReorientedFrom().size();
			KeyMB cur_key = new KeyMB(n_a, n_d, n_r);
			if(stats.get(cur_key)!=null)
				stats.put(cur_key, stats.get(cur_key) + 1.0);
			else
				stats.put(cur_key, 1.0);

			outForAUC.println(test.getInt(0,targetIndex) +", " + probs_p[i] + ", "+ probs_is[i] + ", " + best_kappa);//+ ", " + parents_i_list.toString());
		}

		double auroc = AUC.measure(truth, probs_is);
		double auroc_p = AUC.measure(truth, probs_p);

		System.out.println( "AUROC_P: "+ auroc_p);
		System.out.println( "AUROC: "+ auroc);

		logFile.println( "AUROC_P: "+ auroc_p);
		logFile.println( "AUROC: "+ auroc);		

		for (KeyMB k : stats.keySet()){
			System.out.println(k.print(k) + ":" + (stats.get(k)/testData.getNumRows())*100);
			logFile.println(k.print(k) + ":" + (stats.get(k)/testData.getNumRows())*100);

		}
		System.out.println("-----------------");
		logFile.println("-----------------");


		Map<String, Double> sortedfdist = sortByValue(fdist, false);

		for (String k : sortedfdist.keySet()){
			out.println(k + ", " + (fdist.get(k)/testData.getNumRows()));

		}
		outForAUC.close();
		out.close();
		logFile.close();

	}
	private static double estimation(DataSet trainData, DataSet test, Dag mb, Dag mbt, String target){

		//		List<Node> mbNodes = mb.getNodes();
		//		mbNodes.remove(mb.getNode(target));
		//		System.out.println("mb nodes: " + mbNodes);	
		//		System.out.println("parents:  " + dag.getParents(dag.getNode(target)));	

		double [] probs = classify(mb, mbt, trainData, test, (DiscreteVariable) test.getVariable(target));

		//		BayesPm pm = new BayesPm(dag);
		//		DirichletBayesIm prior = DirichletBayesIm.symmetricDirichletIm(pm, 1.0);
		//		BayesIm im = DirichletEstimator.estimate(prior, trainData);
		//		int targetIndex = im.getNodeIndex(im.getNode(target));
		//		int[] parents = im.getParents(targetIndex);
		//		Arrays.sort(parents);
		//
		//		int[] values = new int[parents.length];
		//		for (int no = 0; no < parents.length; no++){
		//			values [no] = test.getInt(0, parents[no]);
		//		}
		//		double prob = im.getProbability(targetIndex, im.getRowIndex(targetIndex, values), 1);
		//		System.out.println("p_mb: " + probs[1]);
		//		System.out.println("p_pa: " + prob);
		//		return prob;
		return probs[1];
	}

	public static double[] classify(Dag mb, Dag mbt, DataSet train, DataSet test, DiscreteVariable targetVariable) {

		List<Node> mbNodes = mb.getNodes();

		//The Markov blanket nodes will correspond to a subset of the variables
		//in the training dataset.  Find the subset dataset.
		DataSet trainDataSubset = train.subsetColumns(mbNodes);

		//To create a Bayes net for the Markov blanket we need the DAG.
		BayesPm bayesPm = new BayesPm(mb);

		//To parameterize the Bayes net we need the number of values
		//of each variable.
		List<Node> varsTrain = trainDataSubset.getVariables();

		for (int i1 = 0; i1 < varsTrain.size(); i1++) {
			DiscreteVariable trainingVar = (DiscreteVariable) varsTrain.get(i1);
			bayesPm.setCategories(mbNodes.get(i1), trainingVar.getCategories());
		}

		//Create an updater for the instantiated Bayes net.
		DirichletBayesIm prior = DirichletBayesIm.symmetricDirichletIm(bayesPm, 1.0);
		BayesIm bayesIm = DirichletEstimator.estimate(prior, trainDataSubset);

		RowSummingExactUpdater updater = new RowSummingExactUpdater(bayesIm);

		//The subset dataset of the dataset to be classified containing
		//the variables in the Markov blanket.
//		System.out.println("mbNodes: "+ mbNodes);
//		if (!test.getVariables().containsAll(mbNodes)){
//			List<Node> aaa = test.getVariables();
//			aaa.removeAll(mbNodes);
//			System.out.println(aaa);
//		}
//		DataSet testSubset = test;
//		for (String n: test.getVariableNames()){
//			if (mbNodes.(testSubset.ge))
//		}
		DataSet testSubset = test.subsetColumns(mbt.getNodes());

		//Get the raw data from the dataset to be classified, the number
		//of variables, and the number of cases.
		//		int numCases = testSubset.getNumRows();
		double[] estimatedProbs = new double[targetVariable.getNumCategories()];

		//The variables in the dataset.
		List<Node> varsClassify = testSubset.getVariables();

		//For each case in the dataset to be classified compute the estimated
		//value of the target variable and increment the appropriate element
		//of the crosstabulation array.

		//Create an Evidence instance for the instantiated Bayes net
		//which will allow that updating.
		Proposition proposition = Proposition.tautology(bayesIm);

		//Restrict all other variables to their observed values in
		//this case.
		int numMissing = 0;

		for (int testIndex = 0; testIndex < varsClassify.size(); testIndex++) {
			DiscreteVariable var = (DiscreteVariable) varsClassify.get(testIndex);

			// If it's the target, ignore it.
			if (var.equals(targetVariable)) {
				continue;
			}

			int trainIndex = proposition.getNodeIndex(var.getName());

			// If it's not in the train subset, ignore it.
			if (trainIndex == -99) {
				continue;
			}

			int testValue = testSubset.getInt(0, testIndex);

			if (testValue == -99) {
				numMissing++;
			} else {
				proposition.setCategory(trainIndex, testValue);
			}
		}

		Evidence evidence = Evidence.tautology(bayesIm);
		evidence.getProposition().restrictToProposition(proposition);
		updater.setEvidence(evidence);

		// for each possible value of target compute its probability in
		// the updated Bayes net.  Select the value with the highest
		// probability as the estimated getValue.
		int targetIndex = proposition.getNodeIndex(targetVariable.getName());


		for (int category = 0; category < targetVariable.getNumCategories(); category++) {
			double marginal = updater.getMarginal(targetIndex, category);
			estimatedProbs [category] = marginal;
		}

		return estimatedProbs;
	}


	private static Graph learnBNIS(DataSet trainData, DataSet test, double kappa, Graph graphP, double samplePrior){
		// learn the instance-specific model
		ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
		scoreI.setSamplePrior(samplePrior);
		scoreI.setKAddition(kappa);
		scoreI.setKDeletion(kappa);
		scoreI.setKReorientation(kappa);
		ISFges fgesI = new ISFges(scoreI);
		fgesI.setPopulationGraph(graphP);
		fgesI.setInitialGraph(graphP);
		Graph graphI = fgesI.search();
		graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());
		return graphI;
	}

	private static double kappaTuning(String pathToKfold, String dataName, double samplePrior, Graph dagP, String target, int targetIndex){
		//inner LOOCV
//		int kvalues = 9;
		int numFolds = 5;
		double[] kappas = new double[]{0.001, 0.1, 0.3, 0.5, 0.7, 0.9};
		double[] kappa_tune = new double[kappas.length];
		double[] auroc_tune = new double[kappas.length];
		
		
		for (int p = 0; p < kappas.length; p++){	
			System.out.print("\n kappa tunning = ");
			double k_add =  kappas[p];
			double k_delete = k_add;
			double k_reverse = k_add;
			kappa_tune[p]= k_add;
			System.out.print(k_add + ", ");
			double auc = 0.0;
			for (int f = 1; f <= numFolds; f++){
				
				System.out.print(" fold: " + f + ", ");
				DataSet trainData = readData(pathToKfold + dataName + "_train_" + f + ".csv");
				DataSet testData = readData(pathToKfold + dataName + "_test_" + f + ".csv");

				double[] probs_is_tune = new double[testData.getNumRows()];
				int[] truth_tune = new int[testData.getNumRows()];
				
				for (int j = 0; j < testData.getNumRows(); j++){
					DataSet test = testData.subsetRows(new int[]{j});

					// learn the instance-specific model
					ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
					scoreI.setKAddition(k_add);
					scoreI.setKDeletion(k_delete);
					scoreI.setKReorientation(k_reverse);
					scoreI.setSamplePrior(samplePrior);
					ISFges fgesI = new ISFges(scoreI);
					fgesI.setPopulationGraph(dagP);
					fgesI.setInitialGraph(dagP);
					Graph graphI = fgesI.search();
					graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());

					// learn a pop model from data + test
					DagInPatternIterator iteratorI = new DagInPatternIterator(graphI);
					Graph dagI = iteratorI.next();
					dagI = GraphUtils.replaceNodes(dagI, trainData.getVariables());
					Graph dagIt = GraphUtils.replaceNodes(dagI, test.getVariables());
					//				BayesPm pmI = new BayesPm(dagI);
					//				DirichletBayesIm priorI = DirichletBayesIm.symmetricDirichletIm(pmI, 1.0);
					//				BayesIm imI = DirichletEstimator.estimate(priorI, trainData66);
					//				fgesI.setPopulationGraph(dagP);

					truth_tune [j] = test.getInt(0, targetIndex);

					//				double prob_i = estimation(imI, test66, diseaseIndex_i);
					Graph mb_i = GraphUtils.markovBlanketDag(dagI.getNode(target), dagI);
					Graph mb_it = GraphUtils.markovBlanketDag(dagIt.getNode(target), dagIt);
					double prob_i = estimation(trainData, test, (Dag) mb_i, (Dag) mb_it, target);
					probs_is_tune[j] = prob_i; 
				}
				auroc_tune[p] += AUC.measure(truth_tune, probs_is_tune);
			}
			auroc_tune[p] /= numFolds;
		}

		QuickSort.sort(auroc_tune, kappa_tune);

		double best_kappa = kappa_tune[kappas.length-1];
		double best_auroc = auroc_tune[kappas.length-1];
		System.out.println("\n BEST AUROC: "+ best_auroc + ", BEST kappa: " + best_kappa);
		return best_kappa;
	}

	private static Graph BNlearn_pop(DataSet trainDataOrig, double samplePrior, double structurePrior) {
		BDeuScore scoreP = new BDeuScore(trainDataOrig);
		scoreP.setSamplePrior(samplePrior);
		scoreP.setStructurePrior(structurePrior);
		Fges fgesP = new Fges (scoreP);
		fgesP.setSymmetricFirstStep(true);
		Graph graphP = fgesP.search();
		graphP = GraphUtils.replaceNodes(graphP, trainDataOrig.getVariables());
		return graphP;
	}
	//	private static IKnowledge createKnowledge(DataSet trainDataOrig) {
	//		int numVars = trainDataOrig.getNumColumns();
	//		IKnowledge knowledge = new Knowledge2();
	//		int[] tiers = new int[2];
	//		tiers[0] = 0;
	//		tiers[1] = 1;
	//		for (int i=0 ; i< numVars; i++) {
	//			if (!trainDataOrig.getVariable(i).getName().equals("PD1response")){
	//				knowledge.addToTier(0, trainDataOrig.getVariable(i).getName());
	//			}
	//			else{
	//				knowledge.addToTier(1, trainDataOrig.getVariable(i).getName());
	//			}
	//		}
	//		knowledge.setTierForbiddenWithin(0, true);
	//		return knowledge;
	//	}
	private static DataSet readData(String pathToData) {
		Path trainDataFile = Paths.get(pathToData);
		char delimiter = ',';
		VerticalDiscreteTabularDatasetReader trainDataReader = new VerticalDiscreteTabularDatasetFileReader(trainDataFile, DelimiterUtils.toDelimiter(delimiter));
		DataSet trainDataOrig = null;
		try {
			trainDataOrig = (DataSet) DataConvertUtils.toDataModel(trainDataReader.readInData());
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
