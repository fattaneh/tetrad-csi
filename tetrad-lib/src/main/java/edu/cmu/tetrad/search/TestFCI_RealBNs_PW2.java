package edu.cmu.tetrad.search;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//import edu.cmu.tetrad.search.GFci;
import edu.cmu.tetrad.algcomparison.statistic.utils.AdjacencyConfusion;
import edu.cmu.tetrad.algcomparison.statistic.utils.ArrowConfusion;
import edu.cmu.tetrad.bayes.BayesIm;
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
import nu.xom.Builder;
import nu.xom.Document;
import nu.xom.ParsingException;


public class TestFCI_RealBNs_PW2 {
	private static String directory;
	private PrintStream out;
	public static void main(String[] args) {
		// read and process input arguments
		Long seed = 1454147771L;
		String data_path =  "/Users/fattanehjabbari/CCD-Project/CS-BN/experiments_newBSC";
		boolean threshold = true;

		// read and process input arguments
		double alpha = 0.05, cutoff = 0.9, latent = 0.2, prior = 0.5;
		int numCases = 1000, depth = 5;
		String modelName = "Alarm", filePath = "/Users/fattanehjabbari/CCD-Project/CS-BN/experiments_newBSC",
				dataPath = "/Users/fattanehjabbari/CCD-Project/CS-BN/";
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "-c":
				numCases = Integer.parseInt(args[i + 1]);
				break;
			case "-d":
				depth = Integer.parseInt(args[i + 1]);
				break;
			case "-lv":
				latent = Double.parseDouble(args[i + 1]);
				break;
			case "-alpha":
				alpha = Double.parseDouble(args[i + 1]);
				break;
			case "-net":
				modelName = args[i + 1];
				break;
			case "-out":
				filePath = args[i + 1];
				break;
			case "-data":
				dataPath = args[i + 1];
				break;
			case "-pr":
				prior = Double.parseDouble(args[i+1]);
				break;
			}
		}
		System.out.println(Arrays.asList(args));

		// create an instance of class and run an experiment on it
		TestFCI_RealBNs_PW2.directory = dataPath;
		double[] lv = new double[]{0.0, 0.1, 0.2};
		int[] cases = new int[]{10000};//, 1000, 2000};
		for (int c: cases){
			for (double l: lv){
//				for (int i = 0; i < 10; i++){
					latent = l;
					numCases = c;
					TestFCI_RealBNs_PW2 t = new TestFCI_RealBNs_PW2();
					cutoff = 0.5;
					t.test_sim(modelName, alpha, threshold, cutoff, depth, latent, numCases, data_path, seed, prior);

//				}
			}
		}
	}

	public void test_sim(String modelName, double alpha, boolean threshold, double cutoff, int depth, double latent, int numCases, String data_path, long seed, double prior){

		RandomUtil.getInstance().setSeed(seed);
		// get the Bayesian network (graph and parameters) of the given model
		BayesIm im = getBayesIM(modelName);
		System.out.println("im:" + im);
		BayesPm pm = im.getBayesPm();
		Graph trueBN = pm.getDag();
		System.out.println("im.nodes:" + im.getNumNodes());
		System.out.println("dag.nodes:" + trueBN.getNumNodes());
		// set the "numLatentConfounders" percentage of variables to be latent
		int numVars = im.getNumNodes();
		int numEdges = trueBN.getNumEdges();
		int LV = (int) Math.floor(latent * numVars);
		GraphUtils.fixLatents4(LV, trueBN);
		System.out.println("Variables set to be latent:" +getLatents(trueBN));
		
		int numLatents = (int) Math.floor(numVars * latent);
		int numSim = 10;
		System.out.println("# nodes: " + numVars + ", # edges: "+ numEdges + ", # numLatents: "+ numLatents + ", # training: " + numCases);
		double[] arrP = new double[numSim], arrR = new double[numSim], adjP = new double[numSim], adjR = new double[numSim], 
				added = new double[numSim], removed = new double[numSim], reoriented = new double[numSim], shdStrict = new double[numSim], shdLenient = new double[numSim];

		double[] arrPI = new double[numSim], arrRI = new double[numSim], adjPI = new double[numSim], adjRI = new double[numSim], 
				addedI = new double[numSim], removedI = new double[numSim], reorientedI = new double[numSim], shdStrictI = new double[numSim], shdLenientI = new double[numSim];


		try {
			//			File dir = new File(data_path+ "/simulation-Gfci-BDeu-WO/");
			File dir = new File(data_path+ "/simulation-GFci/" +"prior" + prior);
			dir.mkdirs();
			String outputFileName = modelName + "V" + numVars +"-E"+ numEdges +"-L"+ latent + "-N" + numCases + "-Th" + threshold  + "-C" + cutoff +"-prior" + prior +"-Fci" +".csv";

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

			// simulate train and test data from BN
			DataSet fullTrainData = im.simulateData(numCases, true);

			// get the observed part of the data only
			DataSet trainData = DataUtils.restrictToMeasured(fullTrainData);

			// compute true PAG
			final DagToPag2 dagToPag = new DagToPag2(trueBN);
			Graph truePag = dagToPag.convert();
			truePag = GraphUtils.replaceNodes(truePag, trueBN.getNodes());
			System.out.println("true pag:" + truePag.getEdges());

			System.out.println("true dag:" + trueBN.getEdges());
			System.out.println("PAG DONE!!!!");

			// learn the population model
			IndTestProbabilisticBDeu2 indTest_pop = new IndTestProbabilisticBDeu2(trainData, prior);
			indTest_pop.setThreshold(threshold);
			indTest_pop.setCutoff(cutoff);
			BDeuScore scoreP = new BDeuScore(trainData);
			GFci fci_pop = new GFci(indTest_pop, scoreP);
//			fci_pop.setDepth(depth);
			Graph graphP = fci_pop.search();
			System.out.println("new method:" + graphP.getEdges());

		
			// learn the population model with old test
//			indTest_IS indTest_IS = new IndTestProbabilisticBDeu(trainData, prior);

//			indTest_IS.setThreshold(threshold);
//			indTest_IS.setCutoff(cutoff);
			IndTestChiSquare indTest_IS = new IndTestChiSquare(trainData, alpha);
			GFci Fci_IS = new GFci(indTest_IS, scoreP);
//			Fci_IS.setDepth(depth);
			Graph graphI = Fci_IS.search();
			System.out.println("old method:" + graphI.getEdges());


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

		printRes(this.out, "POP old", numSim, arrPI, arrRI, adjPI, adjRI, addedI, removedI, reorientedI, shdStrictI, shdLenientI);
		printRes(this.out,"POP new", numSim, arrP, arrR, adjP, adjR, added, removed, reoriented, shdStrict, shdLenient);
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
	private BayesIm getBayesIM(String type) {
		if ("Alarm".equals(type)) {
			return loadBayesIm("Alarm.xdsl", true);
		} else if ("Hailfinder".equals(type)) {
			return loadBayesIm("Hailfinder.xdsl", false);
		} else if ("Hepar".equals(type)) {
			return loadBayesIm("Hepar2.xdsl", true);
		} else if ("Win95".equals(type)) {
			return loadBayesIm("win95pts.xdsl", false);
		} else if ("Barley".equals(type)) {
			return loadBayesIm("barley.xdsl", false);
		}

		throw new IllegalArgumentException("Not a recogized Bayes IM type.");
	}
	private BayesIm loadBayesIm(String filename, boolean useDisplayNames) {
		try {
			Builder builder = new Builder();
			File dir = new File(this.directory + "/xdsl");
			File file = new File(dir, filename);
			Document document = builder.build(file);
			XdslXmlParser parser = new XdslXmlParser();
			parser.setUseDisplayNames(useDisplayNames);
			return parser.getBayesIm(document.getRootElement());
		} catch (ParsingException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
}
