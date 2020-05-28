package edu.cmu.tetrad.search;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.cmu.tetrad.algcomparison.statistic.utils.AdjacencyConfusionIS;
import edu.cmu.tetrad.algcomparison.statistic.utils.ArrowConfusionIS;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.ISMlBayesIm;
import edu.cmu.tetrad.data.BoxDataSet;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DataUtils;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.graph.EdgeListGraph;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphUtils;
import edu.cmu.tetrad.graph.IndependenceFact;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.graph.NodeType;
import edu.cmu.tetrad.search.Fci;
import edu.cmu.tetrad.search.ISBDeuScore;
import edu.cmu.tetrad.search.IndTestDSep;
import edu.cmu.tetrad.search.IndTestProbabilisticBDeu2;
import edu.cmu.tetrad.search.IndTestProbabilisticISBDeu2;
import edu.cmu.tetrad.search.SearchGraphUtils;
import edu.cmu.tetrad.util.RandomUtil;
import edu.cmu.tetrad.util.TextTable;


public class TestGFCI_IS {
	private PrintStream out;
	public static void main(String[] args) {
		// read and process input arguments
		Long seed = 1454147771L;
		String data_path =  "/Users/fattanehjabbari/CCD-Project/CS-BN/experiments_newBSC_IS";
		boolean threshold = true;
		double alpha = 0, cutoff = 0.5, edgesPerNode = 2.0, latent = 0.2, kappa = 0.5, prior = 0.5;;
		int numVars = 100, numCases = 1000, numTests = 1000, numActualTest = 100, numSim = 10, time = 10, nSim=1;

		System.out.println(Arrays.asList(args));
		for ( int i = 0; i < args.length; i++ ) {   
			switch (args[i]) {
			case "-th":
				threshold = Boolean.parseBoolean(args[i+1]);
				break;	
			case "-alpha":
				alpha = Double.parseDouble(args[i+1]);
				break;
			case "-cutoff":
				cutoff = Double.parseDouble(args[i+1]);
				break;
			case "-kappa":
				kappa = Double.parseDouble(args[i+1]);
				break;
			case "-epn":
				edgesPerNode = Double.parseDouble(args[i+1]);
				break;
			case "-l":
				latent = Double.parseDouble(args[i+1]);
				break;
			case "-pr":
				prior = Double.parseDouble(args[i+1]);
				break;
			case "-v":
				numVars = Integer.parseInt(args[i+1]);
				break;
			case "-test":
				numActualTest = Integer.parseInt(args[i+1]);
				break;
			case "-train":
				numCases = Integer.parseInt(args[i+1]);
				break;
			case "-time":
				time = Integer.parseInt(args[i+1]);
				break;
			case "-sim":
				numSim = Integer.parseInt(args[i+1]);
				break;
			case "-nsim":
				nSim = Integer.parseInt(args[i+1]);
				break;
			case "-seed":
				seed =Long.parseLong(args[i+1]);
				break;
			case "-dir":
				data_path = args[i+1];
				break;
			}
		}

		TestGFCI_IS t = new TestGFCI_IS();
		t.test_sim(nSim,alpha, threshold, cutoff, kappa, numVars, edgesPerNode, latent, numCases, numTests, numActualTest, numSim, data_path, time, seed, prior);
	}
	public void test_sim(int sim ,double alpha, boolean threshold, double cutoff, double kappa, int numVars, double edgesPerNode, double latent, int numCases, int numTests, int numActualTest, int numSim, String data_path, int time, long seed, double prior){

		RandomUtil.getInstance().setSeed(seed + 10 * sim);
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
				
		double[] llI = new double[numSim], llP = new double[numSim], llr = new double[numSim];

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
			File dir = new File(data_path+ "/simulation-Fci/" +"prior-pw-q");

			dir.mkdirs();
//			String outputFileName = "V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + "-Th" + threshold  + "-C" + cutoff +"-kappa" + kappa +"-GFci-BDeu-WO" + sim+".csv";
			String outputFileName = "V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + "-Th" + threshold  + "-C" + cutoff + "-"+ sim+".csv";

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
			Graph trueBN = GraphUtils.randomGraphRandomForwardEdges(vars, numLatents, numEdges, 15, 10, 10, false, true);
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

//			System.out.println("trainData: " + trainData);

			// learn the population model
			System.out.println("begin population search");
			IndTestProbabilisticBDeu2 indTest_pop = new IndTestProbabilisticBDeu2(trainData, prior);
			indTest_pop.setThreshold(threshold);
			indTest_pop.setCutoff(cutoff);
//			IndTestChiSquare indTest_pop = new IndTestChiSquare(trainData, 0.05);
			
//			BDeuScore scoreP = new BDeuScore(trainData);
//			GFci fci_pop = new GFci(indTest_pop, scoreP);
			Fci fci_pop = new Fci(indTest_pop);
			Graph graphP = fci_pop.search();
			graphP = GraphUtils.replaceNodes(graphP, trainData.getVariables());

			IndTestDSep dsepP = new IndTestDSep(graphP);
//			dsepP.setVerbose(true);
//			Fci fciP = new Fci(dsepP);
//			Graph pagP = fciP.search();
//			System.out.println("scorePagP: " + scorePagP);
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
//				IndependenceTest dsep = new IndTestDSep(trueBNI);
				
//				long start = System.currentTimeMillis();
//				long end = start + time * 60 * 1000; // 10 * 60 seconds * 1000 ms/sec
//				Graph truePag = null;
//				Boolean notDone = true;
				
//				while (System.currentTimeMillis() < end & notDone){
//				Fci fci = new Fci(dsep);
//				Graph truePag = fci.search();
				final DagToPag2 dagToPag = new DagToPag2(trueBNI);
		        Graph truePag = dagToPag.convert();
		        truePag = GraphUtils.replaceNodes(truePag, trueBNI.getNodes());
//				notDone=false;
//				}
//				System.out.println(truePag);

//				System.out.println("PAG DONE!!!!");
				
				for (Node n: context.keySet()){
					if (context.get(n)){
						avgcsi[s] += 1;
					}
				}

				// learn the instance-specific model 
				IndTestProbabilisticISBDeu2 indTest_IS = new IndTestProbabilisticISBDeu2(trainData, test, dsepP.getH(), graphP);
//				IndTestProbabilisticISBDeu indTest_IS = new IndTestProbabilisticISBDeu(trainData, test, indTest_pop.getH(), graphP);
				indTest_IS.setThreshold(threshold);
				indTest_IS.setCutoff(cutoff);

				
				ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
				scoreI.setKAddition(kappa);
				scoreI.setKDeletion(kappa);
				scoreI.setKReorientation(kappa);

//				GFci_IS Fci_IS = new GFci_IS(indTest_IS, scoreI, graphP);
				Fci Fci_IS = new Fci(indTest_IS);
//				Fci_IS.setInitialGraph(graphP);
				Graph graphI = Fci_IS.search();
				
				IndTestDSep dsepI = new IndTestDSep(graphI);
//				Fci fciI = new Fci(dsepI);
//				Graph pagI = fciI.search();
				
//				System.out.println("dsepI: " + dsepI.getH().size());
				Map<IndependenceFact, Double> union = new HashMap<>(dsepI.getH());
				union.putAll(dsepP.getH());
//				System.out.println("union: " + union.size());

				double scorePagI = 0.0;//computeScoreI(graphI, union, graphP, dsepP.getH(), trainData, test, prior);
				double scorePagP = 0.0; //computeScoreI(graphP, union, graphP, dsepP.getH(), trainData, test, prior);

				llI[s] += scorePagI;
				llP[s] += scorePagP;
				llr [s] += (scorePagI - scorePagP);
//				System.out.println("dsepP.getH()" + dsepP.getH());
//				System.out.println("dsepI.getH()" + dsepI.getH());
			
	
//				System.out.println("indTest_pop.getH()" + indTest_pop.getH());
//				System.out.println("indTest_IS.getH()" + indTest_IS.getH());
//				
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

			if (arrIPIc != 0){
				arrIPI[s] /= arrIPIc;
			}
			if (arrIRIc != 0){
				arrIRI[s] /= arrIRIc;
			}
			if (arrNPIc != 0){
			arrNPI[s] /= arrNPIc;
			}
			if (arrNRIc != 0){
				arrNRI[s] /= arrNRIc;
			}
			if (arrPIc != 0){
				arrPI[s] /= arrPIc;
			}
			if(arrRIc != 0){
				arrRI[s] /= arrRIc;
			}
			if (adjIPIc != 0){
				adjIPI[s] /= adjIPIc;
			}
			if (adjIRIc != 0){
				adjIRI[s] /= adjIRIc;
			}
			
			if (adjNPIc != 0){
				adjNPI[s] /= adjNPIc;
			}
			if (adjNRIc != 0){
				adjNRI[s] /= adjNRIc;
			}
			if (adjPIc != 0){
				adjPI[s] /= adjPIc;
			}
			if (adjRIc != 0){
				
				adjRI[s] /= adjRIc;
			}
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

			if (arrIPc != 0) {
				arrIP[s] /= arrIPc;
			}
			if(arrIRc != 0) {
				arrIR[s] /= arrIRc;
			}
			if (arrNPc != 0){
				arrNP[s] /= arrNPc;
			}
			if (arrNRc != 0){
				arrNR[s] /= arrNRc;
			}
			if(arrPc != 0){
				arrP[s] /= arrPc;
			}
			if (arrRc != 0){
				arrR[s] /= arrRc;
			}
			if (adjIPc != 0){
				adjIP[s] /= adjIPc;
			}
			if (adjIRc != 0){
				adjIR[s] /= adjIRc;
			}
			if (adjNPc != 0) {
				adjNP[s] /= adjNPc;
			}
			if (adjNRc != 0){
				adjNR[s] /= adjNRc;
			}
			if (adjPc != 0){
				adjP[s] /= adjPc;
			}
			if(adjRc != 0){
				adjR[s] /= adjRc;
			}

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

			llI[s] /= numActualTest;
			llP[s] /= numActualTest;
			llr [s] /= numActualTest;
//			System.out.println("arrIPI: " + Arrays.toString(arrIPI));
//			System.out.println("arrNP: " + Arrays.toString(arrNP));

		}
		
		printRes(this.out, "CSI", numSim, arrIPI, arrNPI, arrPI, arrIRI, arrNRI, arrRI, adjIPI, adjNPI, adjPI, adjIRI, adjNRI, adjRI, addedI, removedI, reorientedI, addedI_IS, removedI_IS, reorientedI_IS, addedI_Other, removedI_Other, reorientedI_Other, shdStrictI, shdLenientI, avgcsi, llI, llr);
		printRes(this.out,"POP", numSim, arrIP, arrNP, arrP, arrIR, arrNR, arrR, adjIP, adjNP, adjP, adjIR, adjNR, adjR, added, removed, reoriented, addedIS, removedIS, reorientedIS, addedOther, removedOther, reorientedOther, shdStrict, shdLenient, avgcsi, llP, llr);
		this.out.close();
		System.out.println("----------------------");


	}

	
//	private double computeScoreP(Graph pag, IndTestDSep dsepP, DataSet data){		
//		double lnQ = 0;
//		IndTestProbabilisticBDeu2 indTest = new IndTestProbabilisticBDeu2(data, 0.5);
//
//		for (IndependenceFact fact : dsepP.getH().keySet()) {
//
//			boolean ind = indTest.isIndependent(fact.getX(), fact.getY(), fact.getZ());
//			double p = indTest.getPosterior();
//			p = pag.isDSeparatedFrom(fact.getX(), fact.getY(), fact.getZ()) ? p : (1.0 - p);
//			
//			if (p < -0.0001 || p > 1.0001 || Double.isNaN(p) || Double.isInfinite(p)) {
//				throw new IllegalArgumentException("p illegally equals " + p);
//			}
//
//			double v = lnQ + Math.log10(p);
//
//			if (Double.isNaN(v) || Double.isInfinite(v)) {
//				continue;
//			}
//
//			lnQ = v;
//		}
//		return lnQ;
//	}

	private double computeScoreI(Graph graphI, Map<IndependenceFact, Double> union, Graph graphP, Map<IndependenceFact, Double> Hpop, DataSet data, DataSet test, double prior){
		
		double scoreI = 0.0;
		
		IndTestProbabilisticISBDeu2 indTestI = new IndTestProbabilisticISBDeu2(data, test, union, graphP);

		List<Node> nodes = graphI.getNodes();
		for (int i = 0; i < nodes.size(); i ++){
			for (int j = i + 1; j < nodes.size(); j ++){
//				System.out.println("nodes i: " + nodes.get(i) + ", j: " + nodes.get(j));
				Map<IndependenceFact, Double> HxyI = indTestI.groupbyXYI(union, 
						graphI.getNode(nodes.get(i).getName()), graphI.getNode(nodes.get(j).getName()));
//				System.out.println("HxyI: " + HxyI.keySet());
				
				Map<IndependenceFact, Double> HxyP = indTestI.groupbyXYP(Hpop, 
						graphP.getNode(nodes.get(i).getName()), graphP.getNode(nodes.get(j).getName()));
//				System.out.println("HxyP: " + HxyP.keySet());
				
				DataSet data_xy = new BoxDataSet((BoxDataSet) data);
				DataSet data_rest = new BoxDataSet((BoxDataSet) data);
				indTestI.splitDatabyXY(data, data_xy, data_rest, HxyI);
//				System.out.println("data_xy: " + data_xy.getNumRows());
//				System.out.println("data_rest: " + data_rest.getNumRows());
					
				if (HxyI.size() > 0 && HxyP.size() > 0){
					
					IndTestProbabilisticISBDeu2 indTestxyI = new IndTestProbabilisticISBDeu2(data_xy, test, HxyP, graphP);
					IndTestProbabilisticBDeu2 indTestxyP = new IndTestProbabilisticBDeu2(data_rest, prior);
				
					for (IndependenceFact f: HxyI.keySet() ){
//						System.out.println("f: " + f);
						indTestxyI.isIndependent(f.getX(), f.getY(), f.getZ());
						double pInd = indTestxyI.getPosterior();
						pInd = graphI.isDSeparatedFrom(f.getX(), f.getY(), f.getZ()) ? pInd : (1.0 - pInd);
						double v = scoreI + Math.log10(pInd);
						if (Double.isNaN(v) || Double.isInfinite(v)) {
							continue;
						}
						scoreI = v;
//						System.out.println("scoreI: " + Math.log10(pInd));
					}
					for (IndependenceFact f: HxyP.keySet()){
//						System.out.println("f: " + f);
						indTestxyP.isIndependent(f.getX(), f.getY(), f.getZ());
						double pInd = indTestxyP.getPosterior();
						pInd = graphP.isDSeparatedFrom(f.getX(), f.getY(), f.getZ()) ? pInd : (1.0 - pInd);
						double v = scoreI + Math.log10(pInd);
						if (Double.isNaN(v) || Double.isInfinite(v)) {
							continue;
						}
						scoreI = v;
//						System.out.println("scoreI: " + scoreI);

					}
				}
				else if(HxyI.size() == 0 && HxyP.size() > 0){
					IndTestProbabilisticBDeu2 indTestxyP = new IndTestProbabilisticBDeu2(data, 0.5);
					for (IndependenceFact f: HxyP.keySet()){
//						System.out.println("f: " + f);
						indTestxyP.isIndependent(f.getX(), f.getY(), f.getZ());
						double pInd = indTestxyP.getPosterior();
						pInd = graphP.isDSeparatedFrom(f.getX(), f.getY(), f.getZ()) ? pInd : (1.0 - pInd);
						double v = scoreI + Math.log10(pInd);
						if (Double.isNaN(v) || Double.isInfinite(v)) {
							continue;
						}
						scoreI = v;
//						System.out.println("scoreI: " + scoreI);
					}
				}
				if (HxyI.size() > 0 && HxyP.size() == 0){
					IndTestProbabilisticISBDeu2 indTestxyI = new IndTestProbabilisticISBDeu2(data_xy, test, HxyP, graphP);
					for (IndependenceFact f: HxyI.keySet() ){
//						System.out.println("f: " + f);
						indTestxyI.isIndependent(f.getX(), f.getY(),
								f.getZ().toArray(new Node[f.getZ().size()]));
						double pInd = indTestxyI.getPosterior();
						pInd = graphI.isDSeparatedFrom(f.getX(), f.getY(), f.getZ()) ? pInd : (1.0 - pInd);
						double v = scoreI + Math.log10(pInd);
						if (Double.isNaN(v) || Double.isInfinite(v)) {
							continue;
						}
						scoreI = v;
//						System.out.println("scoreI: " + Math.log10(pInd));
					}
				}
			}
		}
//		scoreI = Math.pow(10.0, scoreI);
//		System.out.println("scoreI: " + scoreI);
		return scoreI;
	}
	
//	private double computeScoreI(Graph graphI, IndTestDSep dsepI, Graph graphP, IndTestDSep dsepP, DataSet data, DataSet test){
//		
//		double scoreI = 0.0;
//		
//		IndTestProbabilisticISBDeu indTestI = new IndTestProbabilisticISBDeu(data, test, dsepP.getH(), graphP);
//
//		List<Node> nodes = graphI.getNodes();
//		for (int i = 0; i < nodes.size(); i ++){
//			for (int j = i + 1; j < nodes.size(); j ++){
////				System.out.println("nodes i: " + nodes.get(i) + ", j: " + nodes.get(j));
//				Map<IndependenceFact, Double> HxyI = indTestI.groupbyXYI(dsepI.getH(), 
//						graphI.getNode(nodes.get(i).getName()), graphI.getNode(nodes.get(j).getName()));
////				System.out.println("HxyI: " + HxyI.keySet());
//				
//				Map<IndependenceFact, Double> HxyP = indTestI.groupbyXYP(dsepP.getH(), 
//						graphP.getNode(nodes.get(i).getName()), graphP.getNode(nodes.get(j).getName()));
////				System.out.println("HxyP: " + HxyP.keySet());
//				
//				DataSet data_xy = new BoxDataSet((BoxDataSet) data);
//				DataSet data_rest = new BoxDataSet((BoxDataSet) data);
//				indTestI.splitDatabyXY(data, data_xy, data_rest, HxyI);
////				System.out.println("data_xy: " + data_xy.getNumRows());
////				System.out.println("data_rest: " + data_rest.getNumRows());
//					
//				if (HxyI.size() > 0 && HxyP.size() > 0){
//					
//					IndTestProbabilisticISBDeu indTestxyI = new IndTestProbabilisticISBDeu(data_xy, test, HxyP, graphP);
//					IndTestProbabilisticBDeu indTestxyP = new IndTestProbabilisticBDeu(data_rest);
//				
//					for (IndependenceFact f: HxyI.keySet() ){
////						System.out.println("f: " + f);
//						double pInd = indTestxyI.computeIndIS(f.getX(), f.getY(),
//								f.getZ().toArray(new Node[f.getZ().size()]));
//						pInd = graphI.isDSeparatedFrom(f.getX(), f.getY(), f.getZ()) ? pInd : (1.0 - pInd);
//						double v = scoreI + Math.log10(pInd);
//						if (Double.isNaN(v) || Double.isInfinite(v)) {
//							continue;
//						}
//						scoreI = v;
////						System.out.println("scoreI: " + Math.log10(pInd));
//					}
//					for (IndependenceFact f: HxyP.keySet()){
////						System.out.println("f: " + f);
//						indTestxyP.isIndependent(f.getX(), f.getY(), f.getZ());
//						double pInd = indTestxyP.getPosterior();
//						pInd = graphP.isDSeparatedFrom(f.getX(), f.getY(), f.getZ()) ? pInd : (1.0 - pInd);
//						double v = scoreI + Math.log10(pInd);
//						if (Double.isNaN(v) || Double.isInfinite(v)) {
//							continue;
//						}
//						scoreI = v;
////						System.out.println("scoreI: " + scoreI);
//
//					}
//				}
//				else if(HxyI.size() == 0 && HxyP.size() > 0){
//					IndTestProbabilisticBDeu indTestxyP = new IndTestProbabilisticBDeu(data);
//					for (IndependenceFact f: HxyP.keySet()){
////						System.out.println("f: " + f);
//						indTestxyP.isIndependent(f.getX(), f.getY(), f.getZ());
//						double pInd = indTestxyP.getPosterior();
//						pInd = graphP.isDSeparatedFrom(f.getX(), f.getY(), f.getZ()) ? pInd : (1.0 - pInd);
//						double v = scoreI + Math.log10(pInd);
//						if (Double.isNaN(v) || Double.isInfinite(v)) {
//							continue;
//						}
//						scoreI = v;
////						System.out.println("scoreI: " + scoreI);
//					}
//				}
//				if (HxyI.size() > 0 && HxyP.size() == 0){
//					IndTestProbabilisticISBDeu indTestxyI = new IndTestProbabilisticISBDeu(data_xy, test, HxyP, graphP);
//					for (IndependenceFact f: HxyI.keySet() ){
////						System.out.println("f: " + f);
//						double pInd = indTestxyI.computeIndIS(f.getX(), f.getY(),
//								f.getZ().toArray(new Node[f.getZ().size()]));
//						pInd = graphI.isDSeparatedFrom(f.getX(), f.getY(), f.getZ()) ? pInd : (1.0 - pInd);
//						double v = scoreI + Math.log10(pInd);
//						if (Double.isNaN(v) || Double.isInfinite(v)) {
//							continue;
//						}
//						scoreI = v;
////						System.out.println("scoreI: " + Math.log10(pInd));
//					}
//				}
//			}
//		}
////		scoreI = Math.pow(10.0, scoreI);
////		System.out.println("scoreI: " + scoreI);
//		return scoreI;
//	}
//	
	

	private List<Node> createVariables(int numVars) {
		// create variables
		List<Node> vars = new ArrayList<>();
		for (int i = 0; i < numVars; i++) {
			vars.add(new DiscreteVariable("X" + i));
		}
		return vars;
	}
//	private double[] computePrecision(List<Double> p_bsc, List<Double> truth_bsc) {
//		double[] pr = new double[2];
//
//		if(p_bsc.size()!=truth_bsc.size()){
//			System.out.println("Arrays do not have the same size!");
//			return pr;
//		}
//
//		double tp = 0.0, fp = 0.0, fn = 0.0;
//		for (int i = 0; i < p_bsc.size(); i++){
//			if(p_bsc.get(i) >= 0.5 && truth_bsc.get(i) == 1.0){
//				tp += 1;
//			}
//			else if(p_bsc.get(i) >= 0.5 && truth_bsc.get(i) == 0.0){
//				fp += 1;
//			}
//			else if(p_bsc.get(i) < 0.5 && truth_bsc.get(i) == 1.0){
//				fn += 1;
//			}
//		}
//		pr[0] = tp/(tp + fp);
//		pr[1] = tp/(tp + fn);
//		return pr;
//	}
	private void printRes(PrintStream out, String alg, int numSim, double[] arrIPI, double[] arrNPI, double[] arrPI, 
			double[] arrIRI, double[] arrNRI, double[] arrRI, double[] adjIPI, double[] adjNPI, double[] adjPI, 
			double[] adjIRI, double[] adjNRI, double[] adjRI, double[] addedI, double[] removedI, double[] reorientedI, 
			double[] addedI_IS, double[] removedI_IS, double[] reorientedI_IS, double[] addedI_Other, double[] removedI_Other, 
			double[] reorientedI_Other, double[] shdStrictI, double[] shdLenientI, double[] avgcsiI,
			double[] ll, double[] llr){

		NumberFormat nf = new DecimalFormat("0.00");
		//			NumberFormat smallNf = new DecimalFormat("0.00E0");

		TextTable table = new TextTable(numSim+2, 8);
		table.setTabDelimited(true);
		String header = ", adj_P_IS, adj_P_NS, adj_P, adj_R_IS, adj_R_NS, adj_R, arr_P_IS, arr_P_NS, arr_P, arr_R_IS, arr_R_NS, arr_R, added_IS, added_NS, added, removed_IS, removed_NS, removed, reoriented_IS, reoriented_NS, reoriented, shd_strict, shd_lenient, avgCSI, ll, llr";
		table.setToken(0, 0, alg);
		table.setToken(0, 1, header);
		double arrIP = 0.0, arrNP = 0.0, arrP = 0.0, arrIR = 0.0, arrNR = 0.0, arrR = 0.0,
				adjIP = 0.0, adjNP = 0.0, adjP = 0.0, adjIR = 0.0, adjNR = 0.0, adjR = 0.0,
				added = 0.0, removed = 0.0, reoriented = 0.0, shdStrict = 0.0, shdLenient =0.0, avgcsi =0.0,
				addedIS = 0.0, removedIS = 0.0, reorientedIS = 0.0,
				addedNS = 0.0, removedNS = 0.0, reorientedNS = 0.0,
				llAvg = 0.0, llrAvg = 0.0;
		for (int i = 0; i < numSim; i++){
			String res = "," +nf.format(adjIPI[i])+","+nf.format(adjNPI[i])+","+nf.format(adjPI[i])+","+ nf.format(adjIRI[i])+
					","+nf.format(adjNRI[i])+","+nf.format(adjRI[i])+","+
					nf.format(arrIPI[i])+","+nf.format(arrNPI[i])+","+nf.format(arrPI[i])+","+ nf.format(arrIRI[i])+
					","+nf.format(arrNRI[i])+","+nf.format(arrRI[i])+","+
					nf.format(addedI_IS[i])+","+nf.format(addedI_Other[i])+","+nf.format(addedI[i])+","+
					nf.format(removedI_IS[i])+","+nf.format(removedI_Other[i])+","+nf.format(removedI[i])+","+
					nf.format(reorientedI_IS[i])+","+nf.format(reorientedI_Other[i])+","+nf.format(reorientedI[i])+","+ 
					nf.format(shdStrictI[i])+","+nf.format(shdLenientI[i])+","+nf.format(avgcsiI[i])+","+
					nf.format(ll[i])+","+nf.format(llr[i]);
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
			llAvg += ll[i];
			llrAvg = llr[i];
		}
		String res =  ","+nf.format(adjIP/numSim)+","+nf.format(adjNP/numSim)+","+nf.format(adjP/numSim)+","+nf.format(adjIR/numSim)+","+nf.format(adjNR/numSim)+","+nf.format(adjR/numSim)+","+
				nf.format(arrIP/numSim)+","+nf.format(arrNP/numSim)+","+nf.format(arrP/numSim)+","+nf.format(arrIR/numSim)+","+nf.format(arrNR/numSim)+","+nf.format(arrR/numSim)+","+
				nf.format(addedIS/numSim)+","+nf.format(addedNS/numSim)+","+nf.format(added/numSim)+","+
				nf.format(removedIS/numSim)+","+nf.format(removedNS/numSim)+","+nf.format(removed/numSim)+","+
				nf.format(reorientedIS/numSim)+","+nf.format(reorientedNS/numSim)+","+nf.format(reoriented/numSim)+","+
				nf.format(shdStrict/numSim)+","+nf.format(shdLenient/numSim) +","+nf.format(avgcsi/numSim)+","+
				nf.format(llAvg/numSim)+","+nf.format(llrAvg/numSim);
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
}
