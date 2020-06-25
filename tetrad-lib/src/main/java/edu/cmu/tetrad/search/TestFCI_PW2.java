package edu.cmu.tetrad.search;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//import edu.cmu.tetrad.search.GFci;
import edu.cmu.tetrad.algcomparison.statistic.utils.AdjacencyConfusion;
import edu.cmu.tetrad.algcomparison.statistic.utils.ArrowConfusion;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.ISMlBayesIm;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DataUtils;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphUtils;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.graph.NodeType;
//import edu.cmu.tetrad.search.BDeuScore;
import edu.cmu.tetrad.search.Fci;
import edu.cmu.tetrad.search.IndTestProbabilisticBDeu;
import edu.cmu.tetrad.search.SearchGraphUtils;
import edu.cmu.tetrad.util.RandomUtil;
import edu.cmu.tetrad.util.TextTable;


public class TestFCI_PW2 {
	private PrintStream out;
	public static void main(String[] args) {
		// read and process input arguments
		Long seed = 1454147771L;
		String data_path =  "/Users/fattanehjabbari/CCD-Project/CS-BN/experiments_newBSC_Chi2";
		boolean threshold = true;
		double alpha = 0.05, cutoff = 0.5, edgesPerNode = 2.0, latent = 0.2, kappa = 0.5, prior = 0.5;
		int numVars = 11, numCases = 1000, numTests = 1000, numActualTest = 100, numSim = 10, time = 10, nSim=1;

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
		for (int c = 5; c <= 5; c++){
			TestFCI_PW2 t = new TestFCI_PW2();
			cutoff = 0.5;
			t.test_sim(nSim,alpha, threshold, cutoff, kappa, numVars, edgesPerNode, latent, numCases, numTests, numActualTest, numSim, data_path, time, seed, prior);
		}
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
				added = new double[numSim], removed = new double[numSim], reoriented = new double[numSim], shdStrict = new double[numSim], shdLenient = new double[numSim];

		double[] arrPI = new double[numSim], arrRI = new double[numSim], adjPI = new double[numSim], adjRI = new double[numSim], 
				addedI = new double[numSim], removedI = new double[numSim], reorientedI = new double[numSim], shdStrictI = new double[numSim], shdLenientI = new double[numSim];
		
		
		try {
//			File dir = new File(data_path+ "/simulation-Gfci-BDeu-WO/");
			File dir = new File(data_path+ "/simulation-Fci/");

			dir.mkdirs();
//			String outputFileName = "V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + "-Th" + threshold  + "-C" + cutoff +"-kappa" + kappa +"-GFci-BDeu-WO"+".csv";
			String outputFileName = "V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + "-Th" + threshold  + "-C" + cutoff +"-prior" + prior +"-Fci" +".csv";

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

			// get the observed part of the data only
			DataSet trainData = DataUtils.restrictToMeasured(fullTrainData);

			// learn the population model
			IndTestProbabilisticBDeu2 indTest_pop = new IndTestProbabilisticBDeu2(trainData, prior);
			indTest_pop.setThreshold(threshold);
			indTest_pop.setCutoff(cutoff);
//			BDeuScore scoreP = new BDeuScore(trainData);

			Fci fci_pop = new Fci(indTest_pop);
			Graph graphP = fci_pop.search();
			System.out.println("new method:" + graphP.getEdges());

//			// compute statistics
//			double arrIPc = 0.0, arrIRc = 0.0, arrNPc = 0.0, arrNRc = 0.0, arrPc = 0.0, arrRc = 0.0, arrIPIc = 0.0, arrIRIc = 0.0, arrNPIc = 0.0, arrNRIc = 0.0, arrPIc = 0.0, arrRIc = 0.0;
//			double adjIPc = 0.0, adjIRc = 0.0, adjNPc = 0.0, adjNRc = 0.0, adjPc = 0.0, adjRc = 0.0, adjIPIc = 0.0, adjIRIc = 0.0, adjNPIc = 0.0, adjNRIc = 0.0, adjPIc = 0.0, adjRIc = 0.0;

			final DagToPag2 dagToPag = new DagToPag2(trueBN);
			Graph truePag = dagToPag.convert();
			truePag = GraphUtils.replaceNodes(truePag, trueBN.getNodes());
			System.out.println("true pag:" + truePag.getEdges());

			System.out.println("true dag:" + trueBN.getEdges());
			System.out.println("PAG DONE!!!!");

			// learn the population model with old test
//			IndTestProbabilisticBDeu indTest_IS = new IndTestProbabilisticBDeu(trainData, prior);
//			indTest_IS.setThreshold(threshold);
//			indTest_IS.setCutoff(cutoff);
			IndTestChiSquare indTest_IS = new IndTestChiSquare(trainData, alpha);

			Fci Fci_IS = new Fci(indTest_IS);
			Graph graphI = Fci_IS.search();
			System.out.println("old method:" + graphI.getEdges());

			

//			ArrowConfusionIS congI = new ArrowConfusionIS(truePag, GraphUtils.replaceNodes(graphI, trueBNI.getNodes()), context);
//			AdjacencyConfusionIS conAdjGI = new AdjacencyConfusionIS(truePag, GraphUtils.replaceNodes(graphI, trueBNI.getNodes()), context);
			ArrowConfusion congI = new ArrowConfusion(truePag, GraphUtils.replaceNodes(graphI, truePag.getNodes()));
			AdjacencyConfusion conAdjGI = new AdjacencyConfusion(truePag, GraphUtils.replaceNodes(graphI, truePag.getNodes()));

			double den = (congI.getArrowsTp()+congI.getArrowsFp());
			if (den != 0.0){
				arrPI[s] += (congI.getArrowsTp() / den);
			}

			den = (congI.getArrowsTp()+congI.getArrowsFn());
			if (den != 0.0){
				arrRI[s] += (congI.getArrowsTp() / den);
			}

		
			den = (conAdjGI.getAdjTp() + conAdjGI.getAdjFp());
			if (den != 0.0){
				adjPI[s] += (conAdjGI.getAdjTp() / den);
			}

			den = (conAdjGI.getAdjTp() + conAdjGI.getAdjFn());
			if (den != 0.0){
				adjRI[s] += (conAdjGI.getAdjTp() / den);
			}

			ArrowConfusion cong = new ArrowConfusion(truePag, GraphUtils.replaceNodes(graphP, truePag.getNodes()));
			AdjacencyConfusion conAdjG = new AdjacencyConfusion(truePag, GraphUtils.replaceNodes(graphP, truePag.getNodes()));

			System.out.println();
			// population model evaluation
			den = (cong.getArrowsTp() + cong.getArrowsFp());
			if (den != 0.0){
				arrP[s] = (cong.getArrowsTp() / den);
			}

			den = (cong.getArrowsTp() + cong.getArrowsFn());
			if (den != 0.0){
				arrR[s] = (cong.getArrowsTp() / den);
			}

			den = (conAdjG.getAdjTp() + conAdjG.getAdjFp());
			if (den != 0.0){
				adjP[s] = (conAdjG.getAdjTp() / den);
			}

			den = (conAdjG.getAdjTp() + conAdjG.getAdjFn());
			if (den != 0.0){
				adjR[s] = (conAdjG.getAdjTp() / den);
			}
			GraphUtils.GraphComparison cmpI = SearchGraphUtils.getGraphComparison(graphI, truePag, true);
			GraphUtils.GraphComparison cmpP = SearchGraphUtils.getGraphComparison(graphP, truePag, true);
			addedI[s] = cmpI.getEdgesAdded().size();
			removedI[s] = cmpI.getEdgesRemoved().size();
			reorientedI[s] = cmpI.getEdgesReorientedTo().size();
			shdStrictI[s] = cmpI.getShdStrict();
			shdLenientI[s] = cmpI.getShdLenient();

			
			added[s] = cmpP.getEdgesAdded().size();
			removed[s] = cmpP.getEdgesRemoved().size();
			reoriented[s] = cmpP.getEdgesReorientedTo().size();
			shdStrict[s] = cmpP.getShdStrict();
			shdLenient[s] = cmpP.getShdLenient();

			
	
	
	}

	printRes(this.out,  "Chi", numSim, arrPI, arrRI, adjPI, adjRI, addedI, removedI, reorientedI, shdStrictI, shdLenientI);
	printRes(this.out,  "BSC", numSim, arrP, arrR, adjP, adjR, added, removed, reoriented, shdStrict, shdLenient);
	this.out.close();
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
private void printRes(PrintStream out, String alg, int numSim, double[] arrPI, double[] arrRI, 
		double[] adjPI, double[] adjRI, 
		double[] addedI, double[] removedI, double[] reorientedI, 
		double[] shdStrictI, double[] shdLenientI){

	NumberFormat nf = new DecimalFormat("0.00");
	//			NumberFormat smallNf = new DecimalFormat("0.00E0");

	TextTable table = new TextTable(numSim+2, 8);
	table.setTabDelimited(true);
	String header = ", adj_P, adj_R, arr_P, arr_R, added, removed, reoriented, shd_strict, shd_lenient";
	table.setToken(0, 0, alg);
	table.setToken(0, 1, header);
	double arrIP = 0.0, arrNP = 0.0, arrP = 0.0, arrIR = 0.0, arrNR = 0.0, arrR = 0.0,
			adjIP = 0.0, adjNP = 0.0, adjP = 0.0, adjIR = 0.0, adjNR = 0.0, adjR = 0.0,
			added = 0.0, removed = 0.0, reoriented = 0.0, shdStrict = 0.0, shdLenient =0.0, avgcsi =0.0,
			addedIS = 0.0, removedIS = 0.0, reorientedIS = 0.0,
			addedNS = 0.0, removedNS = 0.0, reorientedNS = 0.0,
			llAvg = 0.0, llrAvg = 0.0;
	for (int i = 0; i < numSim; i++){
		String res = "," + nf.format(adjPI[i]) + "," + nf.format(adjRI[i])
				+ "," + nf.format(arrPI[i]) + "," + nf.format(arrRI[i])
				+ "," + nf.format(addedI[i]) + "," + nf.format(removedI[i]) + "," + nf.format(reorientedI[i])
				+ "," + nf.format(shdStrictI[i])+","+nf.format(shdLenientI[i]);
		table.setToken(i+1, 0, ""+(i+1));
		table.setToken(i+1, 1, res);
		
		arrP += arrPI[i];
		arrR += arrRI[i];
		adjP += adjPI[i];
		adjR += adjRI[i];
		added += addedI[i];
		removed += removedI[i];
		reoriented += reorientedI[i];
		shdStrict += shdStrictI[i];
		shdLenient += shdLenientI[i];
	}
	String res =  ","+nf.format(adjP/numSim)+","+nf.format(adjR/numSim)+","+
			nf.format(arrP/numSim)+","+nf.format(arrR/numSim)+","+
			nf.format(added/numSim)+","+
			nf.format(removed/numSim)+","+
			nf.format(reoriented/numSim)+","+
			nf.format(shdStrict/numSim)+","+nf.format(shdLenient/numSim);
	
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
