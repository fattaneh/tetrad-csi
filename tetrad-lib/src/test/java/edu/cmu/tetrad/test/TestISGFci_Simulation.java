package edu.cmu.tetrad.test;


import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


import edu.cmu.tetrad.algcomparison.statistic.utils.AdjacencyConfusionIS;
import edu.cmu.tetrad.algcomparison.statistic.utils.ArrowConfusionIS;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.ISMlBayesIm;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DataUtils;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.data.Knowledge2;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
import edu.cmu.tetrad.util.RandomUtil;
import edu.cmu.tetrad.util.TextTable;
//import nu.xom.Builder;
//import nu.xom.Document;
//import nu.xom.ParsingException;

public class TestISGFci_Simulation {
	private PrintStream out;
	private boolean completeRules = false;
	private boolean threshold = true;
	private double prior = 0.5;
	public static void main(String[] args) {
		int[] numVarss = new int[]{20};
		double[] edgesPerNodes = new double[]{2.0, 4.0, 6.0};
		int[] numCasess = {200, 1000, 5000};
		double LV = 0.2;
		for (int numVars: numVarss){
			for (double edgesPerNode : edgesPerNodes){
				for (int numCases : numCasess){
					TestISGFci_Simulation t = new TestISGFci_Simulation();
					t.testSimulation(numVars,edgesPerNode, LV, numCases);
				}
			}
		}
	}

	public void testSimulation(int numVars, double edgesPerNode, double LV, int numCases){
//		RandomUtil.getInstance().setSeed(1454147770L);

		int numTests = 500;
		int minCat = 2;
		int maxCat = 4;
		int numSim = 10;
		double k_add = 0.9;
		double k_delete = k_add; 
		double k_reverse = k_add; 
		double samplePrior = 1.0;
		int numModels = 1;
		final int numEdges = (int) (numVars * edgesPerNode);
		int numLatents = (int) Math.floor(numVars * LV);

		double avgInDeg2 = 0.0, avgOutDeg2 = 0.0;
		System.out.println("# nodes: " + numVars + ", # edges: "+ numEdges + ", # training: " + numCases + ", # test: "+ numTests);
		System.out.println("k add: " + k_add + ", delete: "+ k_delete + ", reverse: " + k_reverse);
		double[] arrP = new double[numSim], arrR = new double[numSim], adjP = new double[numSim], adjR = new double[numSim], 
				added = new double[numSim], removed = new double[numSim], reoriented = new double[numSim],
				addedIS = new double[numSim], removedIS = new double[numSim], reorientedIS = new double[numSim], 
				addedOther = new double[numSim], removedOther = new double[numSim], reorientedOther = new double[numSim];

		double[] arrPI = new double[numSim], arrRI = new double[numSim], adjPI = new double[numSim], adjRI = new double[numSim], 
				addedI = new double[numSim], removedI = new double[numSim], reorientedI = new double[numSim],
				addedI_IS = new double[numSim], removedI_IS = new double[numSim], reorientedI_IS = new double[numSim], 
				addedI_Other = new double[numSim], removedI_Other = new double[numSim], reorientedI_Other = new double[numSim];

		double[] arrIP = new double[numSim], arrIR = new double[numSim], arrNP = new double[numSim], arrNR = new double[numSim];
		double[] arrIPI = new double[numSim], arrIRI = new double[numSim], arrNPI = new double[numSim], arrNRI = new double[numSim];

		double[] adjIP = new double[numSim], adjIR = new double[numSim], adjNP = new double[numSim], adjNR = new double[numSim];
		double[] adjIPI = new double[numSim], adjIRI = new double[numSim], adjNPI = new double[numSim], adjNRI = new double[numSim];
		double[] avgcsi = new double[numSim];
		double[] shdStrict = new double[numSim], shdLenient = new double[numSim], shdAdjacency = new double[numSim];
		double[] shdStrictI = new double[numSim], shdLenientI = new double[numSim], shdAdjacencyI = new double[numSim];
		try {
			File dir = new File("/Users/fattanehjabbari/CCD-Project/CS-BN/dissertation/simulation-GFCI-kappa"+k_add+"/PESS"+samplePrior+"/");

			dir.mkdirs();
			String outputFileName = "V"+numVars +"-E"+ edgesPerNode + "-N"+ numCases + "-T"+ numTests + "-kappa" + k_add+ "-PESS" + samplePrior+".csv";
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

			RandomUtil.getInstance().setSeed(1454147770L + 1000 * s);

			System.out.println("simulation: " + s);

			List<Node> vars = createVariables(numVars);

			// generate true BN and its parameters
			Graph trueBN = GraphUtils.randomGraphRandomForwardEdges(vars, numLatents, numEdges, 10, 10, 10, false, true);
			System.out.println("Latent variables: " + getLatents(trueBN));
			//			outlog.println("Latent variables: " + getLatents(trueBN));

			BayesPm pm = new BayesPm(trueBN, minCat, maxCat);
			ISMlBayesIm im = new ISMlBayesIm(pm, ISMlBayesIm.RANDOM);
			//System.out.println(im);

			// simulate train and test data from BN
			DataSet fullTrainData = im.simulateData(numCases, true);
			DataSet fullTestData = im.simulateData(numTests, true);

			// get the observed part of the data only
			DataSet trainData = DataUtils.restrictToMeasured(fullTrainData);
			DataSet testData = DataUtils.restrictToMeasured(fullTestData);

			// learn the population model
			BDeuScore scoreP = new BDeuScore(trainData);
			scoreP.setSamplePrior(samplePrior);

			IndTestProbabilisticBDeu2 BSCtest = new IndTestProbabilisticBDeu2(trainData, this.prior );
//			IndTestProbabilisticBDeu BSCtest = new IndTestProbabilisticBDeu(trainData, this.prior );

			BSCtest.setThreshold(this.threshold);


			GFci fgesP = new GFci (BSCtest, scoreP);
			Graph graphP = fgesP.search();
			graphP = GraphUtils.replaceNodes(graphP, trainData.getVariables());


			double csi = 0.0;
			for (int i = 0; i < testData.getNumRows(); i++){
				DataSet test = testData.subsetRows(new int[]{i});
				DataSet fullTest = fullTestData.subsetRows(new int[]{i});

				if (i%100 == 0) {System.out.println(i + " test instances done!");}
				//	System.out.println("test: " + test);

				// obtain the true instance-specific BN
				Map <Node, Boolean> context= new HashMap<Node, Boolean>();
				Graph trueBNI = SearchGraphUtils.patternForDag(new EdgeListGraph(GraphUtils.getISGraph(trueBN, im, fullTest, context)));

				for (Node n: context.keySet()){
					if (context.get(n)){
						avgcsi[s] += 1;
					}
				}

				// get the true underlying PAG
				final DagToPag2 dagToPag = new DagToPag2(trueBNI);
				dagToPag.setCompleteRuleSetUsed(this.completeRules);
				Graph PAG_True = dagToPag.convert();
				PAG_True = GraphUtils.replaceNodes(PAG_True, trueBNI.getNodes());
				//outlog.println("PAG_True: " + PAG_True);

				// learn the instance-specific model
				ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
				scoreI.setKAddition(k_add);
				scoreI.setKDeletion(k_delete);
				scoreI.setKReorientation(k_reverse);
				scoreI.setSamplePrior(samplePrior);
				IndTestProbabilisticISBDeu2 testI = new IndTestProbabilisticISBDeu2(trainData, test, BSCtest.getH(), graphP);
//				IndTestProbabilisticISBDeu testI = new IndTestProbabilisticISBDeu(trainData, test, this.prior);

				GFci_IS fgesI = new GFci_IS(testI, scoreI, fgesP.FgesGraph);
				Graph graphI = fgesI.search();
				graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());

				ArrowConfusionIS congI = new ArrowConfusionIS(PAG_True, GraphUtils.replaceNodes(graphI, PAG_True.getNodes()), context);
				AdjacencyConfusionIS conAdjGI = new AdjacencyConfusionIS(PAG_True, GraphUtils.replaceNodes(graphI, PAG_True.getNodes()), context);

				// arr - nodes w/ CSI
				double denP = (congI.getArrowsITp() + congI.getArrowsIFp());
				double denR = (congI.getArrowsITp() + congI.getArrowsIFn());
				if (denP == 0.0 && denR == 0.0){
					arrIPI[s] += 1.0;
					arrIRI[s] += 1.0;
				}
				if (denP != 0.0){
					arrIPI[s] += (congI.getArrowsITp() / denP);
				}
				if (denR != 0.0){
					arrIRI[s] += (congI.getArrowsITp() / denR);
				}

				// arr - nodes w/o CSI	
				denP = (congI.getArrowsNTp() + congI.getArrowsNFp());
				denR = (congI.getArrowsNTp() + congI.getArrowsNFn());
				if (denP == 0.0 && denR == 0.0){
					arrNPI[s] += 1.0;
					arrNRI[s] += 1.0;
				}
				if (denP != 0.0){
					arrNPI[s] += (congI.getArrowsNTp() / denP);
				}
				if (denR != 0.0){
					arrNRI[s] += (congI.getArrowsNTp() / denR);
				}

				// arr - over all nodes 
				denP = (congI.getArrowsTp()+congI.getArrowsFp());
				denR = (congI.getArrowsTp()+congI.getArrowsFn());
				if (denP == 0.0 && denR == 0.0){
					arrPI[s] += 1.0;
					arrRI[s] += 1.0;
				}
				if (denP != 0.0){
					arrPI[s] += (congI.getArrowsTp() / denP);
				}
				if (denR != 0.0){
					arrRI[s] += (congI.getArrowsTp() / denR);
				}

				// adj - nodes w/ CSI
				denP = (conAdjGI.getAdjITp() + conAdjGI.getAdjIFp());
				denR = (conAdjGI.getAdjITp() + conAdjGI.getAdjIFn());
				if (denP == 0.0 && denR == 0.0){
					adjIPI[s] += 1.0;
					adjIRI[s] += 1.0;
				}
				if (denP != 0.0){
					adjIPI[s] += (conAdjGI.getAdjITp() / denP);
				}
				if(denR != 0.0){
					adjIRI[s] += (conAdjGI.getAdjITp() / denR);
				}

				// adj - nodes w/o CSI
				denP = (conAdjGI.getAdjNTp() + conAdjGI.getAdjNFp());
				denR = (conAdjGI.getAdjNTp() + conAdjGI.getAdjNFn());
				if (denP == 0.0 && denR == 0.0){
					adjNPI[s] += 1.0;
					adjNRI[s] += 1.0;
				}
				if (denP != 0.0){
					adjNPI[s] += (conAdjGI.getAdjNTp() / denP);
				}
				if (denR != 0.0){
					adjNRI[s] += (conAdjGI.getAdjNTp() / denR);
				}

				// adj - over all nodes 
				denP = (conAdjGI.getAdjTp() + conAdjGI.getAdjFp());
				denR = (conAdjGI.getAdjTp() + conAdjGI.getAdjFn());
				if (denP == 0.0 && denR == 0.0){
					adjPI[s] += 1.0;
					adjRI[s] += 1.0;
				}
				if (denP != 0.0){
					adjPI[s] += (conAdjGI.getAdjTp() / denP);
				}
				if (denR != 0.0){
					adjRI[s] += (conAdjGI.getAdjTp() / denR);
				}


				// population model evaluation
				ArrowConfusionIS cong = new ArrowConfusionIS(PAG_True, GraphUtils.replaceNodes(graphP, PAG_True.getNodes()), context);
				AdjacencyConfusionIS conAdjG = new AdjacencyConfusionIS(PAG_True, GraphUtils.replaceNodes(graphP, PAG_True.getNodes()), context);

				// arr - nodes w/ CSI
				denP = (cong.getArrowsITp() + cong.getArrowsIFp());
				denR = (cong.getArrowsITp() + cong.getArrowsIFn());
				if (denP == 0.0 && denR == 0.0){
					arrIP[s] += 1.0;
					arrIR[s] += 1.0;
				}
				if (denP != 0.0){
					arrIP[s] += (cong.getArrowsITp() / denP);
				}
				if(denR != 0.0){
					arrIR[s] += (cong.getArrowsITp() / denR);
				}

				// arr - nodes w/o CSI
				denP = (cong.getArrowsNTp() + cong.getArrowsNFp());
				denR = (cong.getArrowsNTp() + cong.getArrowsNFn());
				if (denP == 0.0 && denR == 0.0){
					arrNP[s] += 1.0;
					arrNR[s] += 1.0;
				}
				if (denP != 0.0){
					arrNP[s] += (cong.getArrowsNTp() / denP);
				}
				if (denR != 0.0){
					arrNR[s] += (cong.getArrowsNTp() / denR);
				}

				// arr - over all nodes 
				denP = (cong.getArrowsTp() + cong.getArrowsFp());
				denR = (cong.getArrowsTp() + cong.getArrowsFn());
				if (denP == 0.0 && denR == 0.0){
					arrP[s] += 1.0;
					arrR[s] += 1.0;
				}
				if (denP != 0.0){
					arrP[s] += (cong.getArrowsTp() / denP);
				}
				if (denR != 0.0){
					arrR[s] += (cong.getArrowsTp() / denR);
				}

				// adj - nodes w/ CSI
				denP = (conAdjG.getAdjITp() + conAdjG.getAdjIFp());
				denR = (conAdjG.getAdjITp() + conAdjG.getAdjIFn());
				if (denP == 0.0 && denR == 0.0){
					adjIP[s] += 1.0;
					adjIR[s] += 1.0;
				}
				if (denP != 0.0){
					adjIP[s] += (conAdjG.getAdjITp() / denP);
				}
				if(denR != 0.0){
					adjIR[s] += (conAdjG.getAdjITp() / denR);
				}

				// adj - nodes w/o CSI
				denP = (conAdjG.getAdjNTp() + conAdjG.getAdjNFp());
				denR = (conAdjG.getAdjNTp() + conAdjG.getAdjNFn());
				if (denP == 0.0 && denR == 0.0){
					adjNP[s] += 1.0;
					adjNR[s] += 1.0;
				}
				if (denP != 0.0){
					adjNP[s] += (conAdjG.getAdjNTp() / denP);
				}
				if (denR != 0.0){
					adjNR[s] += (conAdjG.getAdjNTp() / denR);
				}

				// adj - over all nodes 
				denP = (conAdjG.getAdjTp() + conAdjG.getAdjFp());
				denR = (conAdjG.getAdjTp() + conAdjG.getAdjFn());
				if (denP == 0.0 && denR == 0.0){
					adjP[s] += 1.0;
					adjR[s] += 1.0;
				}
				if (denP != 0.0){
					adjP[s] += (conAdjG.getAdjTp() / denP);
				}
				if (denR != 0.0){
					adjR[s] += (conAdjG.getAdjTp() / denR);
				}

				//				System.out.println("-------------");

				GraphUtils.GraphComparison cmpI = SearchGraphUtils.getGraphComparison(graphI, PAG_True, true);
				GraphUtils.GraphComparison cmpP = SearchGraphUtils.getGraphComparison(graphP, PAG_True, true);
				addedI[s] += cmpI.getEdgesAdded().size();
				removedI[s] += cmpI.getEdgesRemoved().size();
				reorientedI[s] += cmpI.getEdgesReorientedTo().size();
				shdStrictI[s] += cmpI.getShdStrict();
				shdLenientI[s] += cmpI.getShdLenient();
				shdAdjacencyI[s] += cmpI.getEdgesAdded().size() + cmpI.getEdgesRemoved().size();

				added[s] += cmpP.getEdgesAdded().size();
				removed[s] += cmpP.getEdgesRemoved().size();
				reoriented[s] += cmpP.getEdgesReorientedTo().size();
				shdStrict[s] += cmpP.getShdStrict();
				shdLenient[s] += cmpP.getShdLenient();
				shdAdjacency[s] += cmpP.getEdgesAdded().size() + cmpP.getEdgesRemoved().size();

				GraphUtils.GraphComparison cmpI2 = SearchGraphUtils.getGraphComparison(graphI, trueBNI, context);
				GraphUtils.GraphComparison cmpP2 = SearchGraphUtils.getGraphComparison(graphP, trueBNI, context);
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
			avgcsi[s] /= (numVars * numTests);
			System.out.println("avgsci : "+ avgcsi[s]);

			arrIPI[s] /= numTests;
			arrIRI[s] /= numTests;
			arrNPI[s] /= numTests;
			arrNRI[s] /= numTests;
			arrPI[s] /= numTests;
			arrRI[s] /= numTests;
			
			adjIPI[s] /= numTests;
			adjIRI[s] /= numTests;
			adjNPI[s] /= numTests;
			adjNRI[s] /= numTests;
			adjPI[s] /= numTests;
			adjRI[s] /= numTests;

			addedI[s] /= numTests;
			removedI[s] /= numTests;
			reorientedI[s] /= numTests;
			shdStrictI[s] /= numTests;
			shdLenientI[s] /= numTests;
			shdAdjacencyI[s] /= numTests;

			addedI_IS[s] /= numTests;
			removedI_IS[s] /= numTests;
			reorientedI_IS[s] /= numTests;
			addedI_Other[s] /= numTests;
			removedI_Other[s] /= numTests;
			reorientedI_Other[s] /= numTests;

			arrIP[s] /= numTests;
			arrIR[s] /= numTests;
			arrNP[s] /= numTests;
			arrNR[s] /= numTests;
			arrP[s] /= numTests;
			arrR[s] /= numTests;
			
			adjIP[s] /= numTests;
			adjIR[s] /= numTests;
			adjNP[s] /= numTests;
			adjNR[s] /= numTests;
			adjP[s] /= numTests;
			adjR[s] /= numTests;
			
			added[s] /= numTests;
			removed[s] /= numTests;
			reoriented[s] /= numTests;
			shdStrict[s] /= numTests;
			shdLenient[s] /= numTests;
			shdAdjacency[s] /= numTests;

			addedIS[s] /= numTests;
			removedIS[s] /= numTests;
			reorientedIS[s] /= numTests;
			addedOther[s] /= numTests;
			removedOther[s] /= numTests;
			reorientedOther[s] /= numTests;


		}
		//	printRes("CSI", numSim, adjIPI, adjNPI, adjPI, adjIRI, adjNRI, adjRI, addedI, removedI, reorientedI, llr);
		printRes(this.out, "CSI", numSim, arrIPI, arrNPI, arrPI, arrIRI, arrNRI, arrRI, adjIPI, adjNPI, adjPI,
				adjIRI, adjNRI, adjRI, addedI, removedI, reorientedI, addedI_IS, removedI_IS, reorientedI_IS, 
				addedI_Other, removedI_Other, reorientedI_Other, shdStrictI, shdLenientI, shdAdjacencyI);

		//		printRes("POP", numSim, adjIP, adjNP, adjP, adjIR, adjNR, adjR, added, removed, reoriented, llr);
		printRes(this.out,"POP", numSim, arrIP, arrNP, arrP, arrIR, arrNR, arrR, adjIP, adjNP, adjP, adjIR, adjNR,
				adjR, added, removed, reoriented, addedIS, removedIS, reorientedIS, addedOther, removedOther, 
				reorientedOther, shdStrict, shdLenient, shdAdjacency);
		this.out.close();
		System.out.println(avgOutDeg2/numSim);
		System.out.println(avgInDeg2/numSim);
		System.out.println("----------------------");

	}
	private List<Node> createVariables(int numVars) {
		// create variables
		List<Node> vars = new ArrayList<>();
		for (int i = 0; i < numVars; i++) {
			vars.add(new DiscreteVariable("X" + i));
		}
		return vars;
	}
	private void printRes(PrintStream out, String alg, int numSim, double[] arrIPI, double[] arrNPI, 
			double[] arrPI, double[] arrIRI, double[] arrNRI, double[] arrRI, double[] adjIPI, 
			double[] adjNPI, double[] adjPI, double[] adjIRI, double[] adjNRI, double[] adjRI, 
			double[] addedI, double[] removedI, double[] reorientedI, double[] addedI_IS, double[] 
					removedI_IS, double[] reorientedI_IS, double[] addedI_Other, double[] removedI_Other,
					double[] reorientedI_Other, double[] shdStrictI, double[] shdLenientI, double[]shdAdjacencyI){
		NumberFormat nf = new DecimalFormat("0.00");
		//		NumberFormat smallNf = new DecimalFormat("0.00E0");

		TextTable table = new TextTable(numSim+2, 8);
		table.setTabDelimited(true);
		String header = ", adj_P_IS, adj_P_NS, adj_P, adj_R_IS, adj_R_NS, adj_R, arr_P_IS, arr_P_NS, arr_P,"
				+ " arr_R_IS, arr_R_NS, arr_R, added_IS, added_NS, added, removed_IS, removed_NS, removed, "
				+ "reoriented_IS, reoriented_NS, reoriented, S-SHD, L-SHD, A-SHD";
		table.setToken(0, 0, alg);
		table.setToken(0, 1, header);
		double arrIP = 0.0, arrNP = 0.0, arrP = 0.0, arrIR = 0.0, arrNR = 0.0, arrR = 0.0,
				adjIP = 0.0, adjNP = 0.0, adjP = 0.0, adjIR = 0.0, adjNR = 0.0, adjR = 0.0,
				added = 0.0, removed = 0.0, reoriented = 0.0,
				addedIS = 0.0, removedIS = 0.0, reorientedIS = 0.0,
				addedNS = 0.0, removedNS = 0.0, reorientedNS = 0.0, sshd = 0.0, lshd = 0.0, ashd = 0.0;
		for (int i = 0; i < numSim; i++){
			String res = "," +nf.format(adjIPI[i])+","+nf.format(adjNPI[i])+","+nf.format(adjPI[i])+","+ nf.format(adjIRI[i])+
					","+nf.format(adjNRI[i])+","+nf.format(adjRI[i])+","+
					nf.format(arrIPI[i])+","+nf.format(arrNPI[i])+","+nf.format(arrPI[i])+","+ nf.format(arrIRI[i])+
					","+nf.format(arrNRI[i])+","+nf.format(arrRI[i])+","+
					nf.format(addedI_IS[i])+","+nf.format(addedI_Other[i])+","+nf.format(addedI[i])+","+
					nf.format(removedI_IS[i])+","+nf.format(removedI_Other[i])+","+nf.format(removedI[i])+","+
					nf.format(reorientedI_IS[i])+","+nf.format(reorientedI_Other[i])+","+nf.format(reorientedI[i])+","+
					nf.format(shdStrictI[i])+","+nf.format(shdLenientI[i])+","+nf.format(shdAdjacencyI[i]);
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
			addedIS += addedI_IS[i];
			removedIS += removedI_IS[i];
			reorientedIS += reorientedI_IS[i];
			addedNS += addedI_Other[i];
			removedNS += removedI_Other[i];
			reorientedNS += reorientedI_Other[i];
			sshd += shdStrictI[i];
			lshd += shdLenientI[i];
			ashd += shdAdjacencyI[i];
		}
		String res =  ","+nf.format(adjIP/numSim)+","+nf.format(adjNP/numSim)+","+nf.format(adjP/numSim)+","+nf.format(adjIR/numSim)+","+nf.format(adjNR/numSim)+","+nf.format(adjR/numSim)+","+
				nf.format(arrIP/numSim)+","+nf.format(arrNP/numSim)+","+nf.format(arrP/numSim)+","+nf.format(arrIR/numSim)+","+nf.format(arrNR/numSim)+","+nf.format(arrR/numSim)+","+
				nf.format(addedIS/numSim)+","+nf.format(addedNS/numSim)+","+nf.format(added/numSim)+","+
				nf.format(removedIS/numSim)+","+nf.format(removedNS/numSim)+","+nf.format(removed/numSim)+","+
				nf.format(reorientedIS/numSim)+","+nf.format(reorientedNS/numSim)+","+nf.format(reoriented/numSim)+","+
				nf.format(sshd/numSim)+","+nf.format(lshd/numSim)+","+nf.format(ashd/numSim);
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
