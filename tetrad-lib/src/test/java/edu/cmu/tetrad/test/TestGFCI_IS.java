package edu.cmu.tetrad.test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.cmu.tetrad.search.GFci;
import edu.cmu.tetrad.search.GFci_IS;
import edu.cmu.tetrad.algcomparison.statistic.utils.AdjacencyConfusionIS;
import edu.cmu.tetrad.algcomparison.statistic.utils.ArrowConfusionIS;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.ISMlBayesIm;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DataUtils;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.graph.EdgeListGraph;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphUtils;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.graph.NodeType;
import edu.cmu.tetrad.search.BDeuScore;
import edu.cmu.tetrad.search.Fci;
import edu.cmu.tetrad.search.ISBDeuScore;
import edu.cmu.tetrad.search.ISFges;
import edu.cmu.tetrad.search.IndTestDSep;
import edu.cmu.tetrad.search.IndTestProbabilisticBDeu;
import edu.cmu.tetrad.search.IndTestProbabilisticISBDeu;
import edu.cmu.tetrad.search.IndependenceTest;
import edu.cmu.tetrad.search.SearchGraphUtils;
import edu.cmu.tetrad.util.DataConvertUtils;
import edu.cmu.tetrad.util.DelimiterUtils;
import edu.cmu.tetrad.util.RandomUtil;
import edu.cmu.tetrad.util.TextTable;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetFileReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetReader;
import edu.cmu.tetrad.test.*;
//class KeyMB {
//
//	public final int n_a;
//	public final int n_d;
//	public final int n_r;
//
//
//	public KeyMB(final int n_a, final int n_d, final int n_r) {
//		this.n_a = n_a;
//		this.n_d = n_d;
//		this.n_r = n_r;
//	}
//	@Override
//	public boolean equals (final Object O) {
//		if (!(O instanceof KeyMB)) return false;
//		if (((KeyMB) O).n_a != n_a) return false;
//		if (((KeyMB) O).n_d != n_d) return false;
//		if (((KeyMB) O).n_r != n_r) return false;
//		return true;
//	}
//	 @Override
//	 public int hashCode() {
//		 return this.n_a ^ this.n_d ^ this.n_r ;
//	 }
//	 public String print(KeyMB key){
//		return "("+key.n_a +", "+ key.n_d +", "+ key.n_r + ")";
//	 }
//
//}
public class TestGFCI_IS {
	private PrintStream out;
	public static void main(String[] args) {
		// read and process input arguments
		Long seed = 1454147770L;
		String data_path =  System.getProperty("user.dir");
		boolean threshold = true;
		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/UCI/";
		String dataName = "agaricus-lepiota";
		String pathToData = pathToFolder + dataName + ".csv";
		String target = "y";
		
		// Read in the data
		DataSet trainDataOrig = readData(pathToData);
		System.out.println(trainDataOrig.getNumRows() +", " + trainDataOrig.getNumColumns());

		double alpha = 0, cutoff = 0.5, latent = 0.2, kappa = 0.5;
		int numVars = trainDataOrig.getNumColumns(); 
		int numCases = trainDataOrig.getNumRows();
		
		// Create the knowledge

		// learn the population model using all training data
		IndTestProbabilisticBDeu indTest_pop = new IndTestProbabilisticBDeu(trainDataOrig);
		indTest_pop.setThreshold(threshold);
		indTest_pop.setCutoff(cutoff);
		BDeuScore scoreP = new BDeuScore(trainDataOrig);
		GFci fci_pop = new GFci(indTest_pop, scoreP);
		Graph graphP = fci_pop.search();
	
		PrintStream logFile;
		try {
			File dir = new File( pathToFolder + "/outputs/" + dataName);
			dir.mkdirs();
			String outputFileName = dataName  +"_log.txt";
			File fileAUC = new File(dir, outputFileName);
			logFile = new PrintStream(new FileOutputStream(fileAUC));

		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		
		logFile.println(trainDataOrig.getNumRows() +", " + trainDataOrig.getNumColumns());
		logFile.println("Pop graph:" + graphP.getEdges());

		
		for (int p = 5; p <= 5; p++){

			double k_add =  p/10.0; //Math.pow(10, -1.0*p);
			
			double average_llr = 0.0;

			Map <KeyMB, Double> stats= new HashMap<KeyMB, Double>();
			PrintStream outForAUC;
			try {
				File dir = new File( pathToFolder + "/outputs/" + dataName);
				dir.mkdirs();
				String outputFileName = dataName + "-AUROC-Kappa"+ k_add  +".csv";
				File fileAUC = new File(dir, outputFileName);
				outForAUC = new PrintStream(new FileOutputStream(fileAUC));

			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			
			System.out.println("kappa = " + k_add);
			logFile.println("kappa = " + k_add);
			

			outForAUC.println("y, population-FGES, instance-specific-FGES");//, DEGs");
			double added = 0.0, removed = 0.0, reoriented = 0.0, shdStrict = 0.0, shdLenient = 0.0;

			//LOOCV loop
			for (int i = 0; i < trainDataOrig.getNumRows(); i++){
				
				DataSet trainData = trainDataOrig.copy();
				DataSet test = trainDataOrig.subsetRows(new int[]{i});
				trainData.removeRows(new int[]{i});
				
				// learn the instance-specific model
				IndTestProbabilisticISBDeu indTest_IS = new IndTestProbabilisticISBDeu(trainData, test, indTest_pop.getH(), graphP);
				indTest_IS.setThreshold(threshold);
				indTest_IS.setCutoff(cutoff);

				ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
				scoreI.setKAddition(kappa);
				scoreI.setKDeletion(kappa);
				scoreI.setKReorientation(kappa);

				GFci_IS Fci_IS = new GFci_IS(indTest_IS, scoreI, fci_pop.FgesGraph);
				Graph graphI = Fci_IS.search();

				//graph comparison
				GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(graphI, graphP, true);
				added += cmp.getEdgesAdded().size();
				removed += cmp.getEdgesRemoved().size();
				reoriented += cmp.getEdgesReorientedTo().size();
				shdStrict += cmp.getShdStrict();
				shdLenient += cmp.getShdLenient();

				int n_a = cmp.getEdgesAdded().size();
				int n_d = cmp.getEdgesRemoved().size();
				int n_r = cmp.getEdgesReorientedFrom().size();

				KeyMB cur_key = new KeyMB(n_a, n_d, n_r);
				if(stats.get(cur_key)!=null)
					stats.put(cur_key, stats.get(cur_key) + 1.0);
				else
					stats.put(cur_key, 1.0);

			}
		
			for (KeyMB k : stats.keySet()){
				System.out.println(k.print(k) + ":" + (stats.get(k)/trainDataOrig.getNumRows())*100);
				logFile.println(k.print(k) + ":" + (stats.get(k)/trainDataOrig.getNumRows())*100);

			}
			added /= trainDataOrig.getNumRows();
			removed /= trainDataOrig.getNumRows();
			reoriented /= trainDataOrig.getNumRows();
			shdStrict /= trainDataOrig.getNumRows();
			shdLenient /= trainDataOrig.getNumRows();
			System.out.println("average_llr: " + (average_llr/ trainDataOrig.getNumRows()));
			logFile.println("avg added: " + added);
			logFile.println("avg removed: " + removed);
			logFile.println("avg reoriented: " + reoriented);
			logFile.println("avg shdStrict: " + shdStrict);
			logFile.println("avg shdLenient: " + shdLenient);

			System.out.println("-----------------");
			logFile.println("-----------------");
			outForAUC.close();
		}
		logFile.close();

	}
		
	public void test_sim(double alpha, boolean threshold, double cutoff, double kappa, int numVars, double edgesPerNode, double latent, int numCases, int numTests, int numActualTest, int numSim, String data_path, int time, long seed){

		RandomUtil.getInstance().setSeed(seed);
		int minCat = 2;
		int maxCat = 4;
		final int numEdges = (int) (numVars * edgesPerNode);
		int numLatents = (int) Math.floor(numVars * latent);

		System.out.println("# nodes: " + numVars + ", # edges: "+ numEdges + ", # numLatents: "+ numLatents + ", # training: " + numCases + ", # test: "+ numActualTest);
		System.out.println("kappa:  " + kappa);
		double[] arrP = new double[numSim], arrR = new double[numSim], adjP = new double[numSim], adjR = new double[numSim], 
				added = new double[numSim], removed = new double[numSim], reoriented = new double[numSim], shdStrict = new double[numSim], shdLenient = new double[numSim],
				addedIS = new double[numSim], removedIS = new double[numSim], reorientedIS = new double[numSim], 
				addedOther = new double[numSim], removedOther = new double[numSim], reorientedOther = new double[numSim];

		double[] arrPI = new double[numSim], arrRI = new double[numSim], adjPI = new double[numSim], adjRI = new double[numSim], 
				addedI = new double[numSim], removedI = new double[numSim], reorientedI = new double[numSim], shdStrictI = new double[numSim], shdLenientI = new double[numSim],
				addedI_IS = new double[numSim], removedI_IS = new double[numSim], reorientedI_IS = new double[numSim], 
				addedI_Other = new double[numSim], removedI_Other = new double[numSim], reorientedI_Other = new double[numSim];

		double[] arrIP = new double[numSim], arrIR = new double[numSim], arrNP = new double[numSim], arrNR = new double[numSim];
		double[] arrIPI = new double[numSim], arrIRI = new double[numSim], arrNPI = new double[numSim], arrNRI = new double[numSim];

		double[] adjIP = new double[numSim], adjIR = new double[numSim], adjNP = new double[numSim], adjNR = new double[numSim];
		double[] adjIPI = new double[numSim], adjIRI = new double[numSim], adjNPI = new double[numSim], adjNRI = new double[numSim];

		double[] avgcsi = new double[numSim];
		try {
			File dir = new File(data_path+ "/simulation-Gfci-BDeu-WO-/");

			dir.mkdirs();
			String outputFileName = "V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + "-Th" + threshold  + "-C" + cutoff +"-kappa" + kappa +"-GFci-BDeu-WO.csv";
			File file = new File(dir, outputFileName);
			if (file.exists() && file.length() != 0){ 
				return;
			}else{
				this.out = new PrintStream(new FileOutputStream(file));
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

		// loop over simulations
		for (int s = 0; s < numSim; s++){

			System.out.println("simulation: " + s);

			List<Node> vars = createVariables(numVars);

			// generate true BN and its parameters
			Graph trueBN = GraphUtils.randomGraphRandomForwardEdges(vars, numLatents, numEdges, 30, 15, 15, false, true);
			System.out.println("Latent variables: " + getLatents(trueBN));

			System.out.println("generating pm ...");

			BayesPm pm = new BayesPm(trueBN, minCat, maxCat);
			System.out.println("generating im ...");

			ISMlBayesIm im = new ISMlBayesIm(pm, ISMlBayesIm.RANDOM);

			System.out.println("simulating data ...");
			// simulate train and test data from BN
			DataSet fullTrainData = im.simulateData(numCases, true);
			DataSet fullTestData = im.simulateData(numTests, true);

			// get the observed part of the data only
			DataSet trainData = DataUtils.restrictToMeasured(fullTrainData);
			DataSet testData = DataUtils.restrictToMeasured(fullTestData);


			// learn the population model
//			System.out.println("begin population search");
			IndTestProbabilisticBDeu indTest_pop = new IndTestProbabilisticBDeu(trainData);
			indTest_pop.setThreshold(threshold);
			indTest_pop.setCutoff(cutoff);
//			IndTestChiSquare indTest_pop = new IndTestChiSquare(trainData, 0.05);

			BDeuScore scoreP = new BDeuScore(trainData);

			GFci fci_pop = new GFci(indTest_pop, scoreP);
			Graph graphP = fci_pop.search();
			
//			System.out.println("run time (sec): " + end/1000 );

			// compute statistics
			double arrIPc = 0.0, arrIRc = 0.0, arrNPc = 0.0, arrNRc = 0.0, arrPc = 0.0, arrRc = 0.0, arrIPIc = 0.0, arrIRIc = 0.0, arrNPIc = 0.0, arrNRIc = 0.0, arrPIc = 0.0, arrRIc = 0.0;
			double adjIPc = 0.0, adjIRc = 0.0, adjNPc = 0.0, adjNRc = 0.0, adjPc = 0.0, adjRc = 0.0, adjIPIc = 0.0, adjIRIc = 0.0, adjNPIc = 0.0, adjNRIc = 0.0, adjPIc = 0.0, adjRIc = 0.0;
			int caseIndex = 0;
			for (int i = 0; i < numActualTest; i++){
				DataSet test = testData.subsetRows(new int[]{caseIndex});
//				
				if (i%10 == 0) {
					System.out.println(i);
				}
				
				// obtain the true instance-specific BN
				Map <Node, Boolean> context= new HashMap<Node, Boolean>();
				
				DataSet fullTest = fullTestData.subsetRows(new int[]{caseIndex});
				caseIndex ++;

				Graph trueBNI = SearchGraphUtils.patternForDag(new EdgeListGraph(GraphUtils.getISGraph(trueBN, im, fullTest, context)));
				IndependenceTest dsep = new IndTestDSep(trueBNI);
				long start = System.currentTimeMillis();
				long end = start + time * 60 * 1000; // 10 * 60 seconds * 1000 ms/sec
				Graph truePag = null;
				Boolean notDone = true;
				
				while (System.currentTimeMillis() < end & notDone){
					Fci fci = new Fci(dsep);
					truePag = fci.search();
					truePag = GraphUtils.replaceNodes(truePag, trueBNI.getNodes());
					notDone=false;
				}
				
				if (truePag == null){
					break;
				}
				
				for (Node n: context.keySet()){
					if (context.get(n)){
						avgcsi[s] += 1;
					}
				}

				// learn the instance-specific model
				IndTestProbabilisticISBDeu indTest_IS = new IndTestProbabilisticISBDeu(trainData, test, indTest_pop.getH(), graphP);
				indTest_IS.setThreshold(threshold);
				indTest_IS.setCutoff(cutoff);

				ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
				scoreI.setKAddition(kappa);
				scoreI.setKDeletion(kappa);
				scoreI.setKReorientation(kappa);

				GFci_IS Fci_IS = new GFci_IS(indTest_IS, scoreI, fci_pop.FgesGraph);
				Graph graphI = Fci_IS.search();

				ArrowConfusionIS congI = new ArrowConfusionIS(truePag, GraphUtils.replaceNodes(graphI, trueBNI.getNodes()), context);
				AdjacencyConfusionIS conAdjGI = new AdjacencyConfusionIS(truePag, GraphUtils.replaceNodes(graphI, trueBNI.getNodes()), context);

				double den = (congI.getArrowsITp() + congI.getArrowsIFp());
				if (den != 0.0){
					arrIPIc ++;
					arrIPI[s] += (congI.getArrowsITp() / den);
				}

				den = (congI.getArrowsITp() + congI.getArrowsIFn());
				if (den != 0.0){
					arrIRIc ++;
					arrIRI[s] += (congI.getArrowsITp() / den);
				}

				den = (congI.getArrowsNTp() + congI.getArrowsNFp());
				if (den != 0.0){
					arrNPIc ++;
					arrNPI[s] += (congI.getArrowsNTp() / den);
				}

				den = (congI.getArrowsNTp() + congI.getArrowsNFn());
				if (den != 0.0){
					arrNRIc ++;
					arrNRI[s] += (congI.getArrowsNTp() / den);
				}

				den = (congI.getArrowsTp()+congI.getArrowsFp());
				if (den != 0.0){
					arrPIc++;
					arrPI[s] += (congI.getArrowsTp() / den);
				}

				den = (congI.getArrowsTp()+congI.getArrowsFn());
				if (den != 0.0){
					arrRIc ++;
					arrRI[s] += (congI.getArrowsTp() / den);
				}

				den = (conAdjGI.getAdjITp() + conAdjGI.getAdjIFp());
				if (den != 0.0){
					adjIPIc ++;
					adjIPI[s] += (conAdjGI.getAdjITp() / den);
				}

				den = (conAdjGI.getAdjNTp() + conAdjGI.getAdjNFp());
				if (den != 0.0){
					adjNPIc++;
					adjNPI[s] += (conAdjGI.getAdjNTp() / den);
				}

				den = (conAdjGI.getAdjITp() + conAdjGI.getAdjIFn());
				if (den != 0.0){
					adjIRIc  ++;
					adjIRI[s] += (conAdjGI.getAdjITp() / den);
				}

				den = (conAdjGI.getAdjNTp() + conAdjGI.getAdjNFn());
				if (den != 0.0){
					adjNRIc ++;
					adjNRI[s] += (conAdjGI.getAdjNTp() / den);
				}

				den = (conAdjGI.getAdjTp() + conAdjGI.getAdjFp());
				if (den != 0.0){
					adjPIc ++;
					adjPI[s] += (conAdjGI.getAdjTp() / den);
				}

				den = (conAdjGI.getAdjTp() + conAdjGI.getAdjFn());
				if (den != 0.0){
					adjRIc ++;
					adjRI[s] += (conAdjGI.getAdjTp() / den);
				}

				ArrowConfusionIS cong = new ArrowConfusionIS(truePag, GraphUtils.replaceNodes(graphP, truePag.getNodes()), context);
				AdjacencyConfusionIS conAdjG = new AdjacencyConfusionIS(truePag, GraphUtils.replaceNodes(graphP, truePag.getNodes()), context);

				// population model evaluation
				den = (cong.getArrowsITp() + cong.getArrowsIFp());
				if (den != 0.0){
					arrIPc++;
					arrIP[s] += (cong.getArrowsITp() / den);
				}

				den = (cong.getArrowsITp() + cong.getArrowsIFn());
				if(den != 0.0){
					arrIRc ++;
					arrIR[s] += (cong.getArrowsITp() / den);
				}

				den = (cong.getArrowsNTp() + cong.getArrowsNFp());
				if (den != 0.0){
					arrNPc++;
					arrNP[s] += (cong.getArrowsNTp() / den);
				}

				den = (cong.getArrowsNTp() + cong.getArrowsNFn());
				if (den != 0.0){
					arrNRc ++;
					arrNR[s] += (cong.getArrowsNTp() / den);
				}

				den = (cong.getArrowsTp() + cong.getArrowsFp());
				if (den != 0.0){
					arrPc++;
					arrP[s] += (cong.getArrowsTp() / den);
				}

				den = (cong.getArrowsTp() + cong.getArrowsFn());
				if (den != 0.0){
					arrRc ++;
					arrR[s] += (cong.getArrowsTp() / den);
				}

				den = (conAdjG.getAdjITp() + conAdjG.getAdjIFp());
				if (den != 0.0){
					adjIPc++;
					adjIP[s] += (conAdjG.getAdjITp() / den);
				}

				den = (conAdjG.getAdjITp() + conAdjG.getAdjIFn());
				if(den != 0.0){
					adjIRc ++;
					adjIR[s] += (conAdjG.getAdjITp() / den);
				}

				den = (conAdjG.getAdjNTp() + conAdjG.getAdjNFp());
				if (den != 0.0){
					adjNPc++;
					adjNP[s] += (conAdjG.getAdjNTp() / den);
				}
				den = (conAdjG.getAdjNTp() + conAdjG.getAdjNFn());
				if (den != 0.0){
					adjNRc ++;
					adjNR[s] += (conAdjG.getAdjNTp() / den);
				}

				den = (conAdjG.getAdjTp() + conAdjG.getAdjFp());
				if (den != 0.0){
					adjPc++;
					adjP[s] += (conAdjG.getAdjTp() / den);
				}

				den = (conAdjG.getAdjTp() + conAdjG.getAdjFn());
				if (den != 0.0){
					adjRc ++;
					adjR[s] += (conAdjG.getAdjTp() / den);
				}
				//	System.out.println("-------------");
				GraphUtils.GraphComparison cmpI = SearchGraphUtils.getGraphComparison(graphI, truePag, true);
				GraphUtils.GraphComparison cmpP = SearchGraphUtils.getGraphComparison(graphP, truePag, true);
				addedI[s] += cmpI.getEdgesAdded().size();
				removedI[s] += cmpI.getEdgesRemoved().size();
				reorientedI[s] += cmpI.getEdgesReorientedTo().size();
				shdStrictI[s] += cmpI.getShdStrict();
				shdLenientI[s] += cmpI.getShdLenient();

				//
				added[s] += cmpP.getEdgesAdded().size();
				removed[s] += cmpP.getEdgesRemoved().size();
				reoriented[s] += cmpP.getEdgesReorientedTo().size();
				shdStrict[s] += cmpP.getShdStrict();
				shdLenient[s] += cmpP.getShdLenient();

				GraphUtils.GraphComparison cmpI2 = SearchGraphUtils.getGraphComparison(graphI, truePag, context);
				GraphUtils.GraphComparison cmpP2 = SearchGraphUtils.getGraphComparison(graphP, truePag, context);
				addedI_IS[s] += cmpI2.getEdgesAddedIS().size();
				removedI_IS[s] += cmpI2.getEdgesRemovedIS().size();
				reorientedI_IS[s] += cmpI2.getEdgesReorientedToIS().size();
				addedI_Other[s] += cmpI2.getEdgesAddedOther().size();
				removedI_Other[s] += cmpI2.getEdgesRemovedOther().size();
				reorientedI_Other[s] += cmpI2.getEdgesReorientedToOther().size();

				addedIS[s] += cmpP2.getEdgesAddedIS().size();
				removedIS[s] += cmpP2.getEdgesRemovedIS().size();
				reorientedIS[s] += cmpP2.getEdgesReorientedToIS().size();
				addedOther[s] += cmpP2.getEdgesAddedOther().size();
				removedOther[s] += cmpP2.getEdgesRemovedOther().size();
				reorientedOther[s] += cmpP2.getEdgesReorientedToOther().size();

			}
			avgcsi[s] /= (numVars * numActualTest);
			System.out.println("avgsci : "+ avgcsi[s]);

			arrIPI[s] /= arrIPIc;
			arrIRI[s] /= arrIRIc;
			arrNPI[s] /= arrNPIc;
			arrNRI[s] /= arrNRIc;
			arrPI[s] /= arrPIc;
			arrRI[s] /= arrRIc;
			adjIPI[s] /= adjIPIc;
			adjIRI[s] /= adjIRIc;
			adjNPI[s] /= adjNPIc;
			adjNRI[s] /= adjNRIc;
			adjPI[s] /= adjPIc;
			adjRI[s] /= adjRIc;
			addedI[s] /= numActualTest;
			removedI[s] /= numActualTest;
			reorientedI[s] /= numActualTest;
			shdStrictI[s] /= numActualTest;
			shdLenientI[s] /= numActualTest;

			addedI_IS[s] /= numActualTest;
			removedI_IS[s] /= numActualTest;
			reorientedI_IS[s] /= numActualTest;
			addedI_Other[s] /= numActualTest;
			removedI_Other[s] /= numActualTest;
			reorientedI_Other[s] /= numActualTest;

			arrIP[s] /= arrIPc;
			arrIR[s] /= arrIRc;
			arrNP[s] /= arrNPc;
			arrNR[s] /= arrNRc;
			arrP[s] /= arrPc;
			arrR[s] /= arrRc;
			adjIP[s] /= adjIPc;
			adjIR[s] /= adjIRc;
			adjNP[s] /= adjNPc;
			adjNR[s] /= adjNRc;
			adjP[s] /= adjPc;
			adjR[s] /= adjRc;

			added[s] /= numActualTest;
			removed[s] /= numActualTest;
			reoriented[s] /= numActualTest;
			shdStrict[s] /= numActualTest;
			shdLenient[s] /= numActualTest;

			addedIS[s] /= numActualTest;
			removedIS[s] /= numActualTest;
			reorientedIS[s] /= numActualTest;
			addedOther[s] /= numActualTest;
			removedOther[s] /= numActualTest;
			reorientedOther[s] /= numActualTest;


		}
		
		printRes(this.out, "CSI", numSim, arrIPI, arrNPI, arrPI, arrIRI, arrNRI, arrRI, adjIPI, adjNPI, adjPI, adjIRI, adjNRI, adjRI, addedI, removedI, reorientedI, addedI_IS, removedI_IS, reorientedI_IS, addedI_Other, removedI_Other, reorientedI_Other, shdStrictI, shdLenientI, avgcsi);
		printRes(this.out,"POP", numSim, arrIP, arrNP, arrP, arrIR, arrNR, arrR, adjIP, adjNP, adjP, adjIR, adjNR, adjR, added, removed, reoriented, addedIS, removedIS, reorientedIS, addedOther, removedOther, reorientedOther, shdStrict, shdLenient, avgcsi);
		this.out.close();
		System.out.println("----------------------");


	}
	private List<Node> createVariables(int numVars) {
		// create variables
		List<Node> vars = new ArrayList<>();
		int[] tiers = new int[numVars];
		for (int i = 0; i < numVars; i++) {
			vars.add(new DiscreteVariable("X" + i));
		}
		return vars;
	}
	private double[] computePrecision(List<Double> p_bsc, List<Double> truth_bsc) {
		double[] pr = new double[2];

		if(p_bsc.size()!=truth_bsc.size()){
			System.out.println("Arrays do not have the same size!");
			return pr;
		}

		double tp = 0.0, fp = 0.0, fn = 0.0;
		for (int i = 0; i < p_bsc.size(); i++){
			if(p_bsc.get(i) >= 0.5 && truth_bsc.get(i) == 1.0){
				tp += 1;
			}
			else if(p_bsc.get(i) >= 0.5 && truth_bsc.get(i) == 0.0){
				fp += 1;
			}
			else if(p_bsc.get(i) < 0.5 && truth_bsc.get(i) == 1.0){
				fn += 1;
			}
		}
		pr[0] = tp/(tp + fp);
		pr[1] = tp/(tp + fn);
		return pr;
	}
	private void printRes(PrintStream out, String alg, int numSim, double[] arrIPI, double[] arrNPI, double[] arrPI, 
			double[] arrIRI, double[] arrNRI, double[] arrRI, double[] adjIPI, double[] adjNPI, double[] adjPI, 
			double[] adjIRI, double[] adjNRI, double[] adjRI, double[] addedI, double[] removedI, double[] reorientedI, 
			double[] addedI_IS, double[] removedI_IS, double[] reorientedI_IS, double[] addedI_Other, double[] removedI_Other, 
			double[] reorientedI_Other, double[] shdStrictI, double[] shdLenientI, double[] avgcsiI){

		NumberFormat nf = new DecimalFormat("0.00");
		//			NumberFormat smallNf = new DecimalFormat("0.00E0");

		TextTable table = new TextTable(numSim+2, 8);
		table.setTabDelimited(true);
		String header = ", adj_P_IS, adj_P_NS, adj_P, adj_R_IS, adj_R_NS, adj_R, arr_P_IS, arr_P_NS, arr_P, arr_R_IS, arr_R_NS, arr_R, added_IS, added_NS, added, removed_IS, removed_NS, removed, reoriented_IS, reoriented_NS, reoriented, shd_strict, shd_lenient, avgCSI";
		table.setToken(0, 0, alg);
		table.setToken(0, 1, header);
		double arrIP = 0.0, arrNP = 0.0, arrP = 0.0, arrIR = 0.0, arrNR = 0.0, arrR = 0.0,
				adjIP = 0.0, adjNP = 0.0, adjP = 0.0, adjIR = 0.0, adjNR = 0.0, adjR = 0.0,
				added = 0.0, removed = 0.0, reoriented = 0.0, shdStrict = 0.0, shdLenient =0.0, avgcsi =0.0,
				addedIS = 0.0, removedIS = 0.0, reorientedIS = 0.0,
				addedNS = 0.0, removedNS = 0.0, reorientedNS = 0.0;
		for (int i = 0; i < numSim; i++){
			String res = "," +nf.format(adjIPI[i])+","+nf.format(adjNPI[i])+","+nf.format(adjPI[i])+","+ nf.format(adjIRI[i])+
					","+nf.format(adjNRI[i])+","+nf.format(adjRI[i])+","+
					nf.format(arrIPI[i])+","+nf.format(arrNPI[i])+","+nf.format(arrPI[i])+","+ nf.format(arrIRI[i])+
					","+nf.format(arrNRI[i])+","+nf.format(arrRI[i])+","+
					nf.format(addedI_IS[i])+","+nf.format(addedI_Other[i])+","+nf.format(addedI[i])+","+
					nf.format(removedI_IS[i])+","+nf.format(removedI_Other[i])+","+nf.format(removedI[i])+","+
					nf.format(reorientedI_IS[i])+","+nf.format(reorientedI_Other[i])+","+nf.format(reorientedI[i])+","+ 
					nf.format(shdStrictI[i])+","+nf.format(shdLenientI[i])+","+nf.format(avgcsiI[i]);
			table.setToken(i+1, 0, ""+(i+1));
			table.setToken(i+1, 1, res);
			arrIP += arrIPI[i];
			arrNP += arrNPI[i];
			arrP += arrPI[i];
			arrIR += arrIRI[i];
			arrNR += arrNRI[i];
			arrR += arrRI[i];
			adjIP += adjIPI[i];
			adjNP += adjNPI[i];
			adjP += adjPI[i];
			adjIR += adjIRI[i];
			adjNR += adjNRI[i];
			adjR += adjRI[i];
			added += addedI[i];
			removed += removedI[i];
			reoriented += reorientedI[i];
			shdStrict += shdStrictI[i];
			shdLenient += shdLenientI[i];
			avgcsi += avgcsiI[i];

			addedIS += addedI_IS[i];
			removedIS += removedI_IS[i];
			reorientedIS += reorientedI_IS[i];
			addedNS += addedI_Other[i];
			removedNS += removedI_Other[i];
			reorientedNS += reorientedI_Other[i];
		}
		String res =  ","+nf.format(adjIP/numSim)+","+nf.format(adjNP/numSim)+","+nf.format(adjP/numSim)+","+nf.format(adjIR/numSim)+","+nf.format(adjNR/numSim)+","+nf.format(adjR/numSim)+","+
				nf.format(arrIP/numSim)+","+nf.format(arrNP/numSim)+","+nf.format(arrP/numSim)+","+nf.format(arrIR/numSim)+","+nf.format(arrNR/numSim)+","+nf.format(arrR/numSim)+","+
				nf.format(addedIS/numSim)+","+nf.format(addedNS/numSim)+","+nf.format(added/numSim)+","+
				nf.format(removedIS/numSim)+","+nf.format(removedNS/numSim)+","+nf.format(removed/numSim)+","+
				nf.format(reorientedIS/numSim)+","+nf.format(reorientedNS/numSim)+","+nf.format(reoriented/numSim)+","+
				nf.format(shdStrict/numSim)+","+nf.format(shdLenient/numSim) +","+nf.format(avgcsi/numSim);
		table.setToken(numSim+1, 0, "avg");
		table.setToken(numSim+1, 1, res);
		out.println(table);
		System.out.println(table);		
	}
	private List<Node> getLatents(Graph dag) {
		List<Node> latents = new ArrayList<>();
		for (Node n : dag.getNodes()) {
			if (n.getNodeType() == NodeType.LATENT) {
				latents.add(n);
			}
		}
		return latents;
	}
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
}
