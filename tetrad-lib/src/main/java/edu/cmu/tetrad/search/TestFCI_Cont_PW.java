package edu.cmu.tetrad.search;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.cmu.tetrad.algcomparison.graph.RandomForward;
import edu.cmu.tetrad.algcomparison.simulation.SemSimulation;
//import edu.cmu.tetrad.search.GFci;
import edu.cmu.tetrad.algcomparison.statistic.utils.AdjacencyConfusion;
import edu.cmu.tetrad.algcomparison.statistic.utils.ArrowConfusion;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DataUtils;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphUtils;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.graph.NodeType;
//import edu.cmu.tetrad.search.BDeuScore;
import edu.cmu.tetrad.search.Fci;
import edu.cmu.tetrad.search.SearchGraphUtils;
import edu.cmu.tetrad.util.Parameters;
import edu.cmu.tetrad.util.RandomUtil;
import edu.cmu.tetrad.util.TextTable;


public class TestFCI_Cont_PW {
	private PrintStream out;
	public static void main(String[] args) {
		// read and process input arguments
		Long seed = 1454147771L;
		String data_path =  "/Users/fattanehjabbari/CCD-Project/CS-BN/dissertation/BSC/Continuous";
		boolean threshold = true;
		double edgesPerNode = 6.0, latent = 0.2, kappa = 0.5, prior = 0.5;
		int numVars = 10, numCases = 1000, numTests = 1000, numActualTest = 100, numSim = 10, time = 10, nSim=1;

		double[] alpha = new double[] {0.01, 0.001, 0.0001, 0.00001}, cutoff = new double[] {0.3, 0.5, 0.7, 0.9}; 
		System.out.println(Arrays.asList(args));
		for ( int i = 0; i < args.length; i++ ) {   
			switch (args[i]) {
			case "-th":
				threshold = Boolean.parseBoolean(args[i+1]);
				break;	
				//			case "-alpha":
				//				alpha = Double.parseDouble(args[i+1]);
				//				break;
				//			case "-cutoff":
				//				cutoff = Double.parseDouble(args[i+1]);
				//				break;
				//			case "-kappa":
				//				kappa = Double.parseDouble(args[i+1]);
				//				break;
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
				//			case "-time":
				//				time = Integer.parseInt(args[i+1]);
				//				break;
			case "-sim":
				numSim = Integer.parseInt(args[i+1]);
				break;
				//			case "-nsim":
				//				nSim = Integer.parseInt(args[i+1]);
				//				break;
			case "-seed":
				seed =Long.parseLong(args[i+1]);
				break;
			case "-dir":
				data_path = args[i+1];
				break;
			}
		}
		TestFCI_Cont_PW t = new TestFCI_Cont_PW();
		t.test_sim(alpha, cutoff, threshold, numVars, edgesPerNode, latent, numCases, numTests, numActualTest, numSim, data_path, seed, prior);
	}

	public void test_sim(double[] alpha, double[] cutoff, boolean threshold, int numVars, double edgesPerNode, double latent, int numCases, int numTests, int numActualTest, int numSim, String data_path, long seed, double prior){

		RandomUtil.getInstance().setSeed(seed);
		int minCat = 2;
		int maxCat = 4;
		final int numEdges = (int) (numVars * edgesPerNode);
		int numLatents = (int) Math.floor(numVars * latent);		

		// define parameters
		Parameters parameters = new Parameters();
		parameters.set("numRuns", numSim);
		parameters.set("sampleSize", numCases);
		parameters.set("numMeasures", numVars);
		parameters.set("numLatents", numLatents);
		parameters.set("saveLatentVars", true);
		parameters.set("randomizeColumns", true);

		parameters.set("avgDegree", edgesPerNode);
		parameters.set("maxDegree", 15);
		parameters.set("maxIndegree", 10);
		parameters.set("maxOutdegree", 10);

		//        parameters.set("numCategories", 3);
//		parameters.set("minCategories", minCat);
//		parameters.set("maxCategories", maxCat);
		parameters.set("differentGraphs", true);

		parameters.set("varLow", 1.0);
		parameters.set("varHigh", 3.0);
		parameters.set("coefLow", 0.2);
		parameters.set("coefHigh", 0.7);
		parameters.set("meanLow", 0.5);
		parameters.set("meanHigh", 1.5);
		parameters.set("standardize", false);
//		parameters.set("percentDiscrete", 0);
		parameters.set("penaltyDiscount", 1);
		parameters.set("structurePrior", 1); 
		parameters.set("samplePrior", 1);
		parameters.set("discretize", false);
		parameters.set("verbose", false); 
		parameters.set("covSymmetric", true);
		parameters.set("connected", false);

		System.out.println("simulating graphs and data ...");
		SemSimulation simulation = new SemSimulation(new RandomForward());
		simulation.createData(parameters);

		System.out.println("# nodes: " + numVars + ", # edges: "+ numEdges + ", # numLatents: "+ numLatents + ", # training: " + numCases + ", # test: "+ numActualTest);
		try {
			File dir = new File(data_path+ "/simulation-Fci-prior" + prior + "/");
			dir.mkdirs();
			String outputFileName  = "Summary-V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + ".csv";
			File file = new File(dir, outputFileName);
			if (file.exists() && file.length() != 0){ 
				return ;
			}
			else{
				this.out = new PrintStream(new FileOutputStream(file));
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

		double[][] results_cutoff = new double [alpha.length][];
		int ind = 0;
		for (double c : cutoff){
			results_cutoff[ind++] = run_alphaORcutoff(c, false,  threshold, numVars, edgesPerNode, latent, 
					numCases, numSim, data_path, prior, simulation);
		}
		double[][] results_alpha = new double [alpha.length][];
		ind = 0;
		for (double a : alpha){
			results_alpha [ind++] = run_alphaORcutoff(a, true,  threshold, numVars, edgesPerNode, latent, 
					numCases, numSim, data_path, prior, simulation);
		}

		StringBuilder builder = new StringBuilder();
		NumberFormat nf = new DecimalFormat("0.00");

		builder.append("method, adj_P, adj_R, arr_P, arr_R, added, removed, reoriented, shd_strict, shd_lenient\n");
		for(int i = 0; i < results_cutoff.length; i++)//for each row
		{
			builder.append("BSC-SemBic (cutoff = " + cutoff[i] + "),");
			for(int j = 0; j < results_cutoff[i].length; j++)//for each column
			{

				builder.append(nf.format(results_cutoff[i][j]));//append to the output string
				if(j < results_cutoff[i].length - 1)//if this is not the last row element
					builder.append(",");//then add comma (if you don't like commas you can use spaces)
			}
			builder.append("\n");//append new line at the end of the row
		}

		for(int i = 0; i < results_alpha.length; i++)//for each row
		{
			builder.append("RBIT (alpha = " + alpha[i] + "),");

			for(int j = 0; j < results_alpha[i].length; j++)//for each column
			{
				builder.append(nf.format(results_alpha[i][j]));//append to the output string
				if(j < results_alpha[i].length - 1)//if this is not the last row element
					builder.append(",");//then add comma (if you don't like commas you can use spaces)
			}
			builder.append("\n");//append new line at the end of the row
		}
		this.out.print(builder.toString());
		this.out.close();
	}

	private double[] run_alphaORcutoff(double aORc, boolean useAlpha, boolean threshold, int numVars, double edgesPerNode,
			double latent, int numCases, int numSim, String data_path, double prior,
			SemSimulation simulation) {
		double[] arr_p = new double[numSim], arr_r = new double[numSim], adj_p = new double[numSim], adj_r = new double[numSim], 
				added = new double[numSim], removed = new double[numSim], reoriented = new double[numSim], strict_shd = new double[numSim], lenient_shd = new double[numSim];
		PrintStream out;
		try {
			File dir = new File(data_path+ "/simulation-Fci-prior" + prior + "/");

			dir.mkdirs();
			String outputFileName ;
			if (useAlpha){
				System.out.println("alpha" + aORc);
				outputFileName = "V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + "-alpha" + aORc +".csv";
			}
			else{
				System.out.println("cutoff" + aORc);
				outputFileName = "V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + "-cutoff" + + aORc + ".csv";
			}


			File file = new File(dir, outputFileName);
			if (file.exists() && file.length() != 0){ 
				return new double[0];
			}
			else{
				out = new PrintStream(new FileOutputStream(file));
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

		// loop over simulations
		for (int s = 0; s < numSim; s++){

			System.out.println("simulation: " + s);

			Graph trueBN = simulation.getTrueGraph(s);
			final DagToPag2 dagToPag = new DagToPag2(trueBN);
			Graph truePag = dagToPag.convert();
			truePag = GraphUtils.replaceNodes(truePag, trueBN.getNodes());
			System.out.println("true pag:" + truePag.getEdges());

			// simulate train and test data from BN
			DataSet fullTrainData = (DataSet) simulation.getDataModel(s);

			// get the observed part of the data only
			DataSet trainData = DataUtils.restrictToMeasured(fullTrainData);
			Graph graph;
			// learn the model
			if (useAlpha){
				IndTestRBIT indTest = new IndTestRBIT(trainData);
				indTest.setAlpha(aORc);
				Fci fci = new Fci(indTest);
				graph = fci.search();
				System.out.println("RBIT test:" + graph.getEdges());

			}
			else{
				IndTestProbabilisticSemBic indTest = new IndTestProbabilisticSemBic(trainData);
				indTest.setThreshold(threshold);
				indTest.setCutoff(aORc);
				Fci fci = new Fci(indTest);
				graph = fci.search();
				System.out.println("BSC-SemBic test:" + graph.getEdges());

			}

			ArrowConfusion confArr = new ArrowConfusion(truePag, GraphUtils.replaceNodes(graph, truePag.getNodes()));
			AdjacencyConfusion confAdj = new AdjacencyConfusion(truePag, GraphUtils.replaceNodes(graph, truePag.getNodes()));

			System.out.println();
			double den = 0.0;
			den = (confArr.getArrowsTp() + confArr.getArrowsFp());
			if (den != 0.0){
				arr_p[s] = (confArr.getArrowsTp() / den);
			}

			den = (confArr.getArrowsTp() + confArr.getArrowsFn());
			if (den != 0.0){
				arr_r[s] = (confArr.getArrowsTp() / den);
			}

			den = (confAdj.getAdjTp() + confAdj.getAdjFp());
			if (den != 0.0){
				adj_p[s] = (confAdj.getAdjTp() / den);
			}

			den = (confAdj.getAdjTp() + confAdj.getAdjFn());
			if (den != 0.0){
				adj_r[s] = (confAdj.getAdjTp() / den);
			}

			GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(graph, truePag, true);
			added[s] = cmp.getEdgesAdded().size();
			removed[s] = cmp.getEdgesRemoved().size();
			reoriented[s] = cmp.getEdgesReorientedTo().size();
			strict_shd[s] = cmp.getShdStrict();
			lenient_shd[s] = cmp.getShdLenient();

		}
		double[] res;
		if (useAlpha){
			res = printRes(out,  "RNIT test (alpha = " +aORc + ")", numSim, arr_p, arr_r, adj_p, adj_r, added, removed, reoriented, strict_shd, lenient_shd);
		}
		else{
			res = printRes(out,  "SemBic test (cutoff = " + aORc + ")", numSim, arr_p, arr_r, adj_p, adj_r, added, removed, reoriented, strict_shd, lenient_shd);
		}
		out.close();
		System.out.println("----------------------");
		return res;

	}

	private double[] printRes(PrintStream out, String alg, int numSim, double[] arrPI, double[] arrRI, 
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
		double arrP = 0.0, arrR = 0.0,
				adjP = 0.0, adjR = 0.0,
				added = 0.0, removed = 0.0, reoriented = 0.0, shdStrict = 0.0, shdLenient =0.0;
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
		return new double[]{adjP/numSim, adjR/numSim, arrP/numSim, arrR/numSim,
				added/numSim, removed/numSim, reoriented/numSim,
				shdStrict/numSim, shdLenient/numSim};
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



//package edu.cmu.tetrad.search;
//
//import java.io.File;
//import java.io.FileOutputStream;
//import java.io.PrintStream;
//import java.text.DecimalFormat;
//import java.text.NumberFormat;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.List;
//
////import edu.cmu.tetrad.search.GFci;
//import edu.cmu.tetrad.algcomparison.statistic.utils.AdjacencyConfusion;
//import edu.cmu.tetrad.algcomparison.statistic.utils.ArrowConfusion;
//import edu.cmu.tetrad.bayes.BayesPm;
//import edu.cmu.tetrad.bayes.ISMlBayesIm;
//import edu.cmu.tetrad.data.ContinuousVariable;
//import edu.cmu.tetrad.data.DataSet;
//import edu.cmu.tetrad.data.DataUtils;
//import edu.cmu.tetrad.data.DiscreteVariable;
//import edu.cmu.tetrad.graph.Graph;
//import edu.cmu.tetrad.graph.GraphUtils;
//import edu.cmu.tetrad.graph.Node;
//import edu.cmu.tetrad.graph.NodeType;
////import edu.cmu.tetrad.search.BDeuScore;
//import edu.cmu.tetrad.search.Fci;
//import edu.cmu.tetrad.search.IndTestProbabilisticBDeu;
//import edu.cmu.tetrad.search.SearchGraphUtils;
//import edu.cmu.tetrad.sem.LargeScaleSimulation;
//import edu.cmu.tetrad.sem.SemIm;
//import edu.cmu.tetrad.sem.SemPm;
//import edu.cmu.tetrad.util.RandomUtil;
//import edu.cmu.tetrad.util.TextTable;
//
//
//public class TestFCI_Cont_PW {
//	private PrintStream out;
//	public static void main(String[] args) {
//		// read and process input arguments
//		Long seed = 1454147771L;
//		String data_path =  "/Users/fattanehjabbari/CCD-Project/CS-BN/experiments_SemBic";
//		boolean threshold = true;
//		double alpha = 0.05, cutoff = 0.5, edgesPerNode = 6.0, latent = 0.2, kappa = 0.5, prior = 0.5;
//		int numVars = 10, numCases = 1000, numTests = 1000, numActualTest = 100, numSim = 10, time = 10, nSim=1;
//		
//		System.out.println(Arrays.asList(args));
//		for ( int i = 0; i < args.length; i++ ) {   
//			switch (args[i]) {
//			case "-th":
//				threshold = Boolean.parseBoolean(args[i+1]);
//				break;	
//			case "-alpha":
//				alpha = Double.parseDouble(args[i+1]);
//				break;
//			case "-cutoff":
//				cutoff = Double.parseDouble(args[i+1]);
//				break;
//			case "-kappa":
//				kappa = Double.parseDouble(args[i+1]);
//				break;
//			case "-epn":
//				edgesPerNode = Double.parseDouble(args[i+1]);
//				break;
//			case "-l":
//				latent = Double.parseDouble(args[i+1]);
//				break;
//			case "-pr":
//				prior = Double.parseDouble(args[i+1]);
//				break;
//			case "-v":
//				numVars = Integer.parseInt(args[i+1]);
//				break;
//			case "-test":
//				numActualTest = Integer.parseInt(args[i+1]);
//				break;
//			case "-train":
//				numCases = Integer.parseInt(args[i+1]);
//				break;
//			case "-time":
//				time = Integer.parseInt(args[i+1]);
//				break;
//			case "-sim":
//				numSim = Integer.parseInt(args[i+1]);
//				break;
//			case "-nsim":
//				nSim = Integer.parseInt(args[i+1]);
//				break;
//			case "-seed":
//				seed =Long.parseLong(args[i+1]);
//				break;
//			case "-dir":
//				data_path = args[i+1];
//				break;
//			}
//		}
//		TestFCI_Cont_PW t = new TestFCI_Cont_PW();
//		t.test_sim(nSim,alpha, threshold, cutoff, kappa, numVars, edgesPerNode, latent, numCases, numTests, numActualTest, numSim, data_path, time, seed, prior);
//	}
//	
//	public void test_sim(int sim ,double alpha, boolean threshold, double cutoff, double kappa, int numVars, double edgesPerNode, double latent, int numCases, int numTests, int numActualTest, int numSim, String data_path, int time, long seed, double prior){
//
//		RandomUtil.getInstance().setSeed(seed + 10 * sim);
//		int minCat = 2;
//		int maxCat = 4;
//		final int numEdges = (int) (numVars * edgesPerNode);
//		int numLatents = (int) Math.floor(numVars * latent);
//
//		System.out.println("# nodes: " + numVars + ", # edges: "+ numEdges + ", # numLatents: "+ numLatents + ", # training: " + numCases + ", # test: "+ numActualTest);
//		System.out.println("kappa:  " + kappa);
//		double[] arrP = new double[numSim], arrR = new double[numSim], adjP = new double[numSim], adjR = new double[numSim], 
//				added = new double[numSim], removed = new double[numSim], reoriented = new double[numSim], shdStrict = new double[numSim], shdLenient = new double[numSim];
//
//		double[] arrPI = new double[numSim], arrRI = new double[numSim], adjPI = new double[numSim], adjRI = new double[numSim], 
//				addedI = new double[numSim], removedI = new double[numSim], reorientedI = new double[numSim], shdStrictI = new double[numSim], shdLenientI = new double[numSim];
//		
//		
//		try {
////			File dir = new File(data_path+ "/simulation-Gfci-BDeu-WO/");
//			File dir = new File(data_path+ "/simulation-Fci/");
//
//			dir.mkdirs();
////			String outputFileName = "V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + "-Th" + threshold  + "-C" + cutoff +"-kappa" + kappa +"-GFci-BDeu-WO"+".csv";
//			String outputFileName = "V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + "-Th" + threshold  + "-C" + cutoff +"-prior" + prior +"-Fci" +".csv";
//
//			File file = new File(dir, outputFileName);
//			if (file.exists() && file.length() != 0){ 
//				return;
//			}else{
//				this.out = new PrintStream(new FileOutputStream(file));
//			}
//		} catch (Exception e) {
//			throw new RuntimeException(e);
//		}
//
//		// loop over simulations
//		for (int s = 0; s < numSim; s++){
//
//			System.out.println("simulation: " + s);
//
//			List<Node> vars = createVariables(numVars);
//
//			// generate true BN and its parameters
//			Graph trueBN = GraphUtils.randomGraphRandomForwardEdges(vars, numLatents, numEdges, 15, 10, 10, false, true);
//			System.out.println("Latent variables: " + getLatents(trueBN));
//			
//			final DagToPag2 dagToPag = new DagToPag2(trueBN);
//			Graph truePag = dagToPag.convert();
//			truePag = GraphUtils.replaceNodes(truePag, trueBN.getNodes());
//			System.out.println("true pag:" + truePag.getEdges());
//
//			System.out.println("true dag:" + trueBN.getEdges());
//			System.out.println("PAG DONE!!!!");
//			
//
//			System.out.println("generating pm ...");
//
//			SemPm pm = new SemPm(trueBN);
//			
//			System.out.println("generating im ...");
//	        SemIm im = new SemIm(pm);			
//
//			System.out.println("simulating data ...");
//			// simulate train and test data from BN
//			DataSet fullTrainData = im.simulateData(numCases, true);
//
//			// get the observed part of the data only
//			DataSet trainData = DataUtils.restrictToMeasured(fullTrainData);
//
//			// learn the population model
//			IndTestProbabilisticSemBic indTest_pop = new IndTestProbabilisticSemBic(trainData);
//			indTest_pop.setThreshold(threshold);
//			indTest_pop.setCutoff(cutoff);
////			BDeuScore scoreP = new BDeuScore(trainData);
//
//			Fci fci_pop = new Fci(indTest_pop);
//			Graph graphP = fci_pop.search();
//			System.out.println("SemBic test:" + graphP.getEdges());
//
////			// compute statistics
////			double arrIPc = 0.0, arrIRc = 0.0, arrNPc = 0.0, arrNRc = 0.0, arrPc = 0.0, arrRc = 0.0, arrIPIc = 0.0, arrIRIc = 0.0, arrNPIc = 0.0, arrNRIc = 0.0, arrPIc = 0.0, arrRIc = 0.0;
////			double adjIPc = 0.0, adjIRc = 0.0, adjNPc = 0.0, adjNRc = 0.0, adjPc = 0.0, adjRc = 0.0, adjIPIc = 0.0, adjIRIc = 0.0, adjNPIc = 0.0, adjNRIc = 0.0, adjPIc = 0.0, adjRIc = 0.0;
//
//			
//
//			// learn the population model with old test
////			IndTestProbabilisticBDeu indTest_IS = new IndTestProbabilisticBDeu(trainData, prior);
////			indTest_IS.setThreshold(threshold);
////			indTest_IS.setCutoff(cutoff);
//			IndTestFisherZ indTest_IS = new IndTestFisherZ(trainData, alpha);
////			indTest_IS.setThreshold(true);
////			indTest_IS.setCutoff(cutoff);
//			Fci Fci_IS = new Fci(indTest_IS);
//			Graph graphI = Fci_IS.search();
//			System.out.println("Fisher test:" + graphI.getEdges());
//
//			
//
////			ArrowConfusionIS congI = new ArrowConfusionIS(truePag, GraphUtils.replaceNodes(graphI, trueBNI.getNodes()), context);
////			AdjacencyConfusionIS conAdjGI = new AdjacencyConfusionIS(truePag, GraphUtils.replaceNodes(graphI, trueBNI.getNodes()), context);
//			ArrowConfusion congI = new ArrowConfusion(truePag, GraphUtils.replaceNodes(graphI, truePag.getNodes()));
//			AdjacencyConfusion conAdjGI = new AdjacencyConfusion(truePag, GraphUtils.replaceNodes(graphI, truePag.getNodes()));
//
//			double den = (congI.getArrowsTp()+congI.getArrowsFp());
//			if (den != 0.0){
//				arrPI[s] += (congI.getArrowsTp() / den);
//			}
//
//			den = (congI.getArrowsTp()+congI.getArrowsFn());
//			if (den != 0.0){
//				arrRI[s] += (congI.getArrowsTp() / den);
//			}
//
//		
//			den = (conAdjGI.getAdjTp() + conAdjGI.getAdjFp());
//			if (den != 0.0){
//				adjPI[s] += (conAdjGI.getAdjTp() / den);
//			}
//
//			den = (conAdjGI.getAdjTp() + conAdjGI.getAdjFn());
//			if (den != 0.0){
//				adjRI[s] += (conAdjGI.getAdjTp() / den);
//			}
//
//			ArrowConfusion cong = new ArrowConfusion(truePag, GraphUtils.replaceNodes(graphP, truePag.getNodes()));
//			AdjacencyConfusion conAdjG = new AdjacencyConfusion(truePag, GraphUtils.replaceNodes(graphP, truePag.getNodes()));
//
//			System.out.println();
//			// population model evaluation
//			den = (cong.getArrowsTp() + cong.getArrowsFp());
//			if (den != 0.0){
//				arrP[s] = (cong.getArrowsTp() / den);
//			}
//
//			den = (cong.getArrowsTp() + cong.getArrowsFn());
//			if (den != 0.0){
//				arrR[s] = (cong.getArrowsTp() / den);
//			}
//
//			den = (conAdjG.getAdjTp() + conAdjG.getAdjFp());
//			if (den != 0.0){
//				adjP[s] = (conAdjG.getAdjTp() / den);
//			}
//
//			den = (conAdjG.getAdjTp() + conAdjG.getAdjFn());
//			if (den != 0.0){
//				adjR[s] = (conAdjG.getAdjTp() / den);
//			}
//			GraphUtils.GraphComparison cmpI = SearchGraphUtils.getGraphComparison(graphI, truePag, true);
//			GraphUtils.GraphComparison cmpP = SearchGraphUtils.getGraphComparison(graphP, truePag, true);
//			addedI[s] = cmpI.getEdgesAdded().size();
//			removedI[s] = cmpI.getEdgesRemoved().size();
//			reorientedI[s] = cmpI.getEdgesReorientedTo().size();
//			shdStrictI[s] = cmpI.getShdStrict();
//			shdLenientI[s] = cmpI.getShdLenient();
//
//			
//			added[s] = cmpP.getEdgesAdded().size();
//			removed[s] = cmpP.getEdgesRemoved().size();
//			reoriented[s] = cmpP.getEdgesReorientedTo().size();
//			shdStrict[s] = cmpP.getShdStrict();
//			shdLenient[s] = cmpP.getShdLenient();
//
//			
//	
//	
//	}
//
//	printRes(this.out,  "SemBic", numSim, arrPI, arrRI, adjPI, adjRI, addedI, removedI, reorientedI, shdStrictI, shdLenientI);
//	printRes(this.out,  "FisherZ", numSim, arrP, arrR, adjP, adjR, added, removed, reoriented, shdStrict, shdLenient);
//	this.out.close();
//	System.out.println("----------------------");
//
//
//}
//
//
//private List<Node> createVariables(int numVars) {
//	// create variables
//	List<Node> vars = new ArrayList<>();
//	for (int i = 0; i < numVars; i++) {
//		vars.add(new ContinuousVariable("X" + i));
//	}
//	return vars;
//}
////	private double[] computePrecision(List<Double> p_bsc, List<Double> truth_bsc) {
////		double[] pr = new double[2];
////
////		if(p_bsc.size()!=truth_bsc.size()){
////			System.out.println("Arrays do not have the same size!");
////			return pr;
////		}
////
////		double tp = 0.0, fp = 0.0, fn = 0.0;
////		for (int i = 0; i < p_bsc.size(); i++){
////			if(p_bsc.get(i) >= 0.5 && truth_bsc.get(i) == 1.0){
////				tp += 1;
////			}
////			else if(p_bsc.get(i) >= 0.5 && truth_bsc.get(i) == 0.0){
////				fp += 1;
////			}
////			else if(p_bsc.get(i) < 0.5 && truth_bsc.get(i) == 1.0){
////				fn += 1;
////			}
////		}
////		pr[0] = tp/(tp + fp);
////		pr[1] = tp/(tp + fn);
////		return pr;
////	}
//private void printRes(PrintStream out, String alg, int numSim, double[] arrPI, double[] arrRI, 
//		double[] adjPI, double[] adjRI, 
//		double[] addedI, double[] removedI, double[] reorientedI, 
//		double[] shdStrictI, double[] shdLenientI){
//
//	NumberFormat nf = new DecimalFormat("0.00");
//	//			NumberFormat smallNf = new DecimalFormat("0.00E0");
//
//	TextTable table = new TextTable(numSim+2, 8);
//	table.setTabDelimited(true);
//	String header = ", adj_P, adj_R, arr_P, arr_R, added, removed, reoriented, shd_strict, shd_lenient";
//	table.setToken(0, 0, alg);
//	table.setToken(0, 1, header);
//	double arrIP = 0.0, arrNP = 0.0, arrP = 0.0, arrIR = 0.0, arrNR = 0.0, arrR = 0.0,
//			adjIP = 0.0, adjNP = 0.0, adjP = 0.0, adjIR = 0.0, adjNR = 0.0, adjR = 0.0,
//			added = 0.0, removed = 0.0, reoriented = 0.0, shdStrict = 0.0, shdLenient =0.0, avgcsi =0.0,
//			addedIS = 0.0, removedIS = 0.0, reorientedIS = 0.0,
//			addedNS = 0.0, removedNS = 0.0, reorientedNS = 0.0,
//			llAvg = 0.0, llrAvg = 0.0;
//	for (int i = 0; i < numSim; i++){
//		String res = "," + nf.format(adjPI[i]) + "," + nf.format(adjRI[i])
//				+ "," + nf.format(arrPI[i]) + "," + nf.format(arrRI[i])
//				+ "," + nf.format(addedI[i]) + "," + nf.format(removedI[i]) + "," + nf.format(reorientedI[i])
//				+ "," + nf.format(shdStrictI[i])+","+nf.format(shdLenientI[i]);
//		table.setToken(i+1, 0, ""+(i+1));
//		table.setToken(i+1, 1, res);
//		
//		arrP += arrPI[i];
//		arrR += arrRI[i];
//		adjP += adjPI[i];
//		adjR += adjRI[i];
//		added += addedI[i];
//		removed += removedI[i];
//		reoriented += reorientedI[i];
//		shdStrict += shdStrictI[i];
//		shdLenient += shdLenientI[i];
//	}
//	String res =  ","+nf.format(adjP/numSim)+","+nf.format(adjR/numSim)+","+
//			nf.format(arrP/numSim)+","+nf.format(arrR/numSim)+","+
//			nf.format(added/numSim)+","+
//			nf.format(removed/numSim)+","+
//			nf.format(reoriented/numSim)+","+
//			nf.format(shdStrict/numSim)+","+nf.format(shdLenient/numSim);
//	
//	table.setToken(numSim+1, 0, "avg");
//	table.setToken(numSim+1, 1, res);
//	out.println(table);
//	System.out.println(table);		
//}
//private List<Node> getLatents(Graph dag) {
//	List<Node> latents = new ArrayList<>();
//	for (Node n : dag.getNodes()) {
//		if (n.getNodeType() == NodeType.LATENT) {
//			latents.add(n);
//		}
//	}
//	return latents;
//}
//}
