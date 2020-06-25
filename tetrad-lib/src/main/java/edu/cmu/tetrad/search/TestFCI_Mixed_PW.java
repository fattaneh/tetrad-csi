package edu.cmu.tetrad.search;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.cmu.tetrad.algcomparison.graph.RandomForward;
import edu.cmu.tetrad.algcomparison.simulation.ConditionalGaussianSimulation;
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


public class TestFCI_Mixed_PW {
	private PrintStream out;
	public static void main(String[] args) {
		// read and process input arguments
		Long seed = 1454147771L;
		String data_path =  "/Users/fattanehjabbari/CCD-Project/CS-BN/experiments_Mixed";
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
		TestFCI_Mixed_PW t = new TestFCI_Mixed_PW();
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
        parameters.set("percentDiscrete", 50);
        
       parameters.set("randomizeColumns", true);

        parameters.set("avgDegree", edgesPerNode);
        parameters.set("maxDegree", 15);
        parameters.set("maxIndegree", 10);
        parameters.set("maxOutdegree", 10);
        
//        parameters.set("numCategories", 3);
        parameters.set("minCategories", minCat);
        parameters.set("maxCategories", maxCat);
        parameters.set("differentGraphs", true);

        parameters.set("varLow", 1.0);
        parameters.set("varHigh", 3.0);
        parameters.set("coefLow", 0.2);
        parameters.set("coefHigh", 0.7);
        parameters.set("meanLow", 0.5);
        parameters.set("meanHigh", 1.5);
        
        parameters.set("percentDiscrete", 50);
        parameters.set("penaltyDiscount", 1);
        parameters.set("structurePrior", 1); 
        parameters.set("samplePrior", 1);
        parameters.set("discretize", false);
        parameters.set("verbose", false); 
        parameters.set("covSymmetric", true);
        parameters.set("connected", false);
        
		System.out.println("simulating graphs and data ...");
        ConditionalGaussianSimulation simulation = new ConditionalGaussianSimulation(new RandomForward());
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
			builder.append("BSC-DG (cutoff = " + cutoff[i] + "),");
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
			builder.append("LR-DG (alpha = " + alpha[i] + "),");

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
			ConditionalGaussianSimulation simulation) {
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
				IndTestDegenerateGaussianLRT indTest = new IndTestDegenerateGaussianLRT(trainData);
				indTest.setAlpha(aORc);
				Fci fci = new Fci(indTest);
				graph = fci.search();
				System.out.println("DGLR test:" + graph.getEdges());

			}
			else{
				IndTestProbabilisticDGScore indTest = new IndTestProbabilisticDGScore(trainData);
				indTest.setThreshold(threshold);
				indTest.setCutoff(aORc);
				Fci fci = new Fci(indTest);
				graph = fci.search();
				System.out.println("DG test:" + graph.getEdges());

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
			res = printRes(out,  "DGLR test (alpha = " +aORc + ")", numSim, arr_p, arr_r, adj_p, adj_r, added, removed, reoriented, strict_shd, lenient_shd);
		}
		else{
			res = printRes(out,  "DG test (cutoff = " + aORc + ")", numSim, arr_p, arr_r, adj_p, adj_r, added, removed, reoriented, strict_shd, lenient_shd);
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
