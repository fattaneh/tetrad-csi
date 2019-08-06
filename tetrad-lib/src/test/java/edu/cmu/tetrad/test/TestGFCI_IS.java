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

import edu.cmu.tetrad.search.GFci;
import edu.cmu.tetrad.search.GFci_IS;
import edu.cmu.tetrad.algcomparison.statistic.utils.AdjacencyConfusionIS;
import edu.cmu.tetrad.algcomparison.statistic.utils.ArrowConfusionIS;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.ISMlBayesIm;
import edu.cmu.tetrad.bayes.MlBayesIm;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DataUtils;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.data.Knowledge2;
import edu.cmu.tetrad.graph.EdgeListGraph;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphUtils;
import edu.cmu.tetrad.graph.IndependenceFact;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.graph.NodeType;
import edu.cmu.tetrad.search.BDeuScore;
import edu.cmu.tetrad.search.Fci;
import edu.cmu.tetrad.search.ISBDeuScore;
import edu.cmu.tetrad.search.IndTestDSep;
import edu.cmu.tetrad.search.IndTestProbabilistic;
import edu.cmu.tetrad.search.IndTestProbabilisticIS;
import edu.cmu.tetrad.search.IndependenceTest;
import edu.cmu.tetrad.search.SearchGraphUtils;
import edu.cmu.tetrad.util.RandomUtil;
import edu.cmu.tetrad.util.TextTable;


public class TestGFCI_IS {
	private PrintStream out;
	public static void main(String[] args) {

		TestGFCI_IS t = new TestGFCI_IS();
		t.test_sim();
	}
	public void test1(){
		int numVars = 4;
		int numCases = 10000;
		int minCat = 2;
		int maxCat = 2;

		List<Node> vars = new ArrayList<>();
		for (int i = 0; i < numVars; i++) {
			vars.add(new DiscreteVariable("X" + i));
		}

		Graph dag = new EdgeListGraph(vars);
		dag.addDirectedEdge(dag.getNode("X0"), dag.getNode("X2"));
		dag.addDirectedEdge(dag.getNode("X1"), dag.getNode("X2"));
		dag.addDirectedEdge(dag.getNode("X2"), dag.getNode("X3"));
		dag.addDirectedEdge(dag.getNode("X1"), dag.getNode("X3"));
		BayesPm pm = new BayesPm(dag, minCat, maxCat);
		MlBayesIm im = new MlBayesIm(pm, MlBayesIm.MANUAL);
		im.setProbability(0, 0, 0, 0.75);
		im.setProbability(0, 0, 1, 0.25);
		im.setProbability(1, 0, 0, 0.62);
		im.setProbability(1, 0, 1, 0.38);
		im.setProbability(2, 0, 0, 0.92);
		im.setProbability(2, 1, 0, 0.92);
		im.setProbability(2, 2, 0, 0.31);
		im.setProbability(2, 3, 0, 0.65);
		im.setProbability(2, 0, 1, 0.08);
		im.setProbability(2, 1, 1, 0.08);
		im.setProbability(2, 2, 1, 0.69);
		im.setProbability(2, 3, 1, 0.35);
		im.setProbability(3, 0, 0, 0.6);
		im.setProbability(3, 1, 0, 0.25);
		im.setProbability(3, 2, 0, 0.9);
		im.setProbability(3, 3, 0, 0.9);
		im.setProbability(3, 0, 1, 0.4);
		im.setProbability(3, 1, 1, 0.75);
		im.setProbability(3, 2, 1, 0.1);
		im.setProbability(3, 3, 1, 0.1);

		DataSet data = im.simulateData(numCases, false);
		DataSet test = im.simulateData(1, false);
		test.setDouble(0, 0, 0);
		test.setDouble(0, 1, 1);
		test.setDouble(0, 2, 1);
		test.setDouble(0, 3, 1);


		System.out.println("-------------------- pop search --------------------");
		double cutoff = 0.9;
		IndTestProbabilistic indTest_pop = new IndTestProbabilistic(data);
		indTest_pop.setThreshold(true);
		indTest_pop.setCutoff(cutoff);
		Fci fci_pop = new Fci(indTest_pop);
		Graph graph_pop = fci_pop.search();
		System.out.println("indTest.getH() pop : " + 		indTest_pop.getH());
		System.out.println(graph_pop);

		System.out.println("-------------------- IS search --------------------");
		IndTestProbabilisticIS indTest_IS = new IndTestProbabilisticIS(data, test, indTest_pop.getH(), graph_pop);
		indTest_IS.setThreshold(true);
		indTest_IS.setCutoff(cutoff);
		ISBDeuScore scoreI = new ISBDeuScore(data, test);
		scoreI.setKAddition(0.1);
		scoreI.setKDeletion(0.1);
		scoreI.setKReorientation(0.1);
		GFci_IS Fci_IS = new GFci_IS(indTest_IS, scoreI, graph_pop);
		Graph graph_IS = Fci_IS.search();
		System.out.println("indTest.getH() is  : " + indTest_IS.getH());		
		System.out.println(graph_IS);
	}
	public void test_sim(){

		RandomUtil.getInstance().setSeed(1454147770L);
		int[] numVarss = new int[]{10};
		double[] edgesPerNodes = new double[]{5.0};
		int numCases = 10000;
		int numTests = 20;
		int minCat = 2;
		int maxCat = 3;
		int numSim = 10;
		boolean threshold = true;
		double cutoff = 0.5;
		double latent = 0.2;	
		List<Double> p_bsc_is = new ArrayList<Double>(); 
		List<Double> p_bsc_pop = new ArrayList<Double>(); 
		List<Double> truth_bsc_is = new ArrayList<Double>(); 
		List<Double> truth_bsc_pop = new ArrayList<Double>(); 

		double[] kappas = new double[] {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
		for (double kappa: kappas){
			double k_add = kappa;
			double k_delete = kappa; 
			double k_reverse = kappa; 
			for (int numVars: numVarss){
				for (double edgesPerNode : edgesPerNodes){
					final int numEdges = (int) (numVars * edgesPerNode);
					double avgInDeg2 = 0.0, avgOutDeg2 = 0.0;
					int numLatents = (int) Math.floor(numVars * latent);

					System.out.println("# nodes: " + numVars + ", # edges: "+ numEdges + ", # numLatents: "+ numLatents + ", # training: " + numCases + ", # test: "+ numTests);
					System.out.println("k add: " + k_add + ", delete: "+ k_delete + ", reverse: " + k_reverse);
					double[] arrP = new double[numSim], arrR = new double[numSim], adjP = new double[numSim], adjR = new double[numSim], 
							added = new double[numSim], removed = new double[numSim], reoriented = new double[numSim],
							addedIS = new double[numSim], removedIS = new double[numSim], reorientedIS = new double[numSim], 
							addedOther = new double[numSim], removedOther = new double[numSim], reorientedOther = new double[numSim], llr = new double[numSim];

					double[] arrPI = new double[numSim], arrRI = new double[numSim], adjPI = new double[numSim], adjRI = new double[numSim], 
							addedI = new double[numSim], removedI = new double[numSim], reorientedI = new double[numSim],
							addedI_IS = new double[numSim], removedI_IS = new double[numSim], reorientedI_IS = new double[numSim], 
							addedI_Other = new double[numSim], removedI_Other = new double[numSim], reorientedI_Other = new double[numSim];

					double[] arrIP = new double[numSim], arrIR = new double[numSim], arrNP = new double[numSim], arrNR = new double[numSim];
					double[] arrIPI = new double[numSim], arrIRI = new double[numSim], arrNPI = new double[numSim], arrNRI = new double[numSim];

					double[] adjIP = new double[numSim], adjIR = new double[numSim], adjNP = new double[numSim], adjNR = new double[numSim];
					double[] adjIPI = new double[numSim], adjIRI = new double[numSim], adjNPI = new double[numSim], adjNRI = new double[numSim];
					double[] avgcsi = new double[numSim];
					try {
						File dir = new File("/Users/fattanehjabbari/CCD-Project/CS-BN/simulation-GFci-IS/");

						dir.mkdirs();
//						String outputFileName = "V"+numVars +"-E"+ edgesPerNode +"-L"+ latent  +"-kappa" + k_add+"-GFci.csv";
			            String outputFileName = "V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + "-T" + threshold +"-kappa" + k_add +"-GFci.csv";

						File file = new File(dir, outputFileName);
						this.out = new PrintStream(new FileOutputStream(file));
					} catch (Exception e) {
						throw new RuntimeException(e);
					}

					// loop over simulations
					for (int s = 0; s < numSim; s++){

						System.out.println("simulation: " + s);

						// create variables
						IKnowledge knowledge = new Knowledge2();
						List<Node> vars = new ArrayList<>();
						int[] tiers = new int[numVars];
						for (int i = 0; i < numVars; i++) {
							vars.add(new DiscreteVariable("X" + i));
							tiers[i] = i;
							knowledge.addToTier(i, "X" + i);
						}

						// generate true BN and its parameters
						Graph trueBN = GraphUtils.randomGraphRandomForwardEdges(vars, numLatents, numEdges, 30, 15, 15, false, true);
						System.out.println(getLatents(trueBN));
						//					for (Node nod: trueBN.getNodes()){
						//						if (trueBN.getIndegree(nod)>1)
						//							avgInDeg2 = avgInDeg2 + 1;
						//						if (trueBN.getOutdegree(nod)>1)
						//							avgOutDeg2  = avgOutDeg2 +1;
						//					}

						BayesPm pm = new BayesPm(trueBN, minCat, maxCat);
						ISMlBayesIm im = new ISMlBayesIm(pm, ISMlBayesIm.RANDOM);

						// simulate train and test data from BN
						DataSet fullTrainData = im.simulateData(numCases, true, tiers);
						DataSet fullTestData = im.simulateData(numTests, true, tiers);


						DataSet trainData = DataUtils.restrictToMeasured(fullTrainData);
						DataSet testData = DataUtils.restrictToMeasured(fullTestData);

						// learn the population model
						IndTestProbabilistic indTest_pop = new IndTestProbabilistic(trainData);
						indTest_pop.setThreshold(threshold);
						indTest_pop.setCutoff(cutoff);
						BDeuScore scoreP = new BDeuScore(trainData);
						GFci fci_pop = new GFci(indTest_pop, scoreP);
						Graph graphP = fci_pop.search();
//						System.out.println("graphP: " + graphP);

						// compute statistics
						double arrIRc = 0.0, arrNRc = 0.0, arrRc = 0.0, arrIRIc = 0.0, arrNRIc = 0.0, arrRIc = 0.0;
						double adjIRc = 0.0, adjNRc = 0.0, adjRc = 0.0, adjIRIc = 0.0, adjNRIc = 0.0, adjRIc = 0.0;
						double csi = 0.0;
						for (int i = 0; i < testData.getNumRows(); i++){
							DataSet test = testData.subsetRows(new int[]{i});
							if (i%100 == 0) {System.out.println(i);}

							// obtain the true instance-specific BN
							Map <Node, Boolean> context= new HashMap<Node, Boolean>();
							DataSet fullTest = fullTestData.subsetRows(new int[]{i});

							Graph trueBNI = SearchGraphUtils.patternForDag(new EdgeListGraph(GraphUtils.getISGraph(trueBN, im, fullTest, context)));
							IndependenceTest dsep = new IndTestDSep(trueBNI);
							Fci fci = new Fci(dsep);
							Graph truePag = fci.search();
							truePag = GraphUtils.replaceNodes(truePag, trueBNI.getNodes());
							
								Map<IndependenceFact, Double> tests_pop = indTest_pop.getH();
								for(IndependenceFact f: tests_pop.keySet()){
									p_bsc_pop.add(tests_pop.get(f));
									List<Node> _z = new ArrayList<Node>();
									for(Node nz : f.getZ()){
										_z.add(truePag.getNode(nz.getName()));
									}
									if (dsep.isIndependent(truePag.getNode(f.getX().getName()), truePag.getNode(f.getY().getName()), _z)){
										truth_bsc_pop.add(1.0);
									}
									else{
										truth_bsc_pop.add(0.0);
									}
								}
							
//							System.out.println("truePag IS: " + truePag);
//							IndependenceTest dsep2 = new IndTestDSep(trueBN);
//							fci = new Fci(dsep2);
//							Graph truePag_pop = fci.search();
//							truePag_pop = GraphUtils.replaceNodes(truePag_pop, trueBN.getNodes());
////							System.out.println("truePag POP: " + truePag_pop);	
//							GraphComparison cmp=  SearchGraphUtils.getGraphComparison(truePag, truePag_pop);
//							System.out.println("num edges pag pop: " + truePag_pop.getNumEdges());
//							System.out.println("num edges pag is : " + truePag_pop.getNumEdges());
//							System.out.println("added: " +cmp.getEdgesAdded().size() + ", removed: " + cmp.getEdgesRemoved().size() + ", reoriented: " + cmp.getEdgesReorientedFrom().size());
							
							for (Node n: context.keySet()){
								if (context.get(n)){
									avgcsi[s] += 1;
								}
							}

							// learn the instance-specific model
							IndTestProbabilisticIS indTest_IS = new IndTestProbabilisticIS(trainData, test, indTest_pop.getH(), graphP);
							indTest_IS.setThreshold(threshold);
							indTest_IS.setCutoff(cutoff);
							ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
							scoreI.setKAddition(kappa);
							scoreI.setKDeletion(kappa);
							scoreI.setKReorientation(kappa);
							GFci_IS Fci_IS = new GFci_IS(indTest_IS, scoreI, graphP);
							Graph graphI = Fci_IS.search();
							//	System.out.println("graphI: " + graphI);
							//	System.out.println("truePag: " + truePag);

							Map<IndependenceFact, Double> tests_is = indTest_IS.getH();
							for(IndependenceFact f: tests_is.keySet()){
								p_bsc_is.add(tests_is.get(f));
								List<Node> _z = new ArrayList<Node>();
								for(Node nz : f.getZ()){
									_z.add(truePag.getNode(nz.getName()));
								}
								if (dsep.isIndependent(truePag.getNode(f.getX().getName()), truePag.getNode(f.getY().getName()), _z)){
									truth_bsc_is.add(1.0);
								}
								else{
									truth_bsc_is.add(0.0);
								}
							}
							ArrowConfusionIS congI = new ArrowConfusionIS(truePag, GraphUtils.replaceNodes(graphI, trueBNI.getNodes()), context);
							AdjacencyConfusionIS conAdjGI = new AdjacencyConfusionIS(truePag, GraphUtils.replaceNodes(graphI, trueBNI.getNodes()), context);

							double den = (congI.getArrowsITp() + congI.getArrowsIFp());
							if (den != 0.0){
								arrIPI[s] += (congI.getArrowsITp() / den);
							}

							den = (congI.getArrowsITp() + congI.getArrowsIFn());
							if (den != 0.0){
								arrIRIc ++;
								arrIRI[s] += (congI.getArrowsITp() / den);
							}

							den = (congI.getArrowsNTp() + congI.getArrowsNFp());
							if (den != 0.0){
								arrNPI[s] += (congI.getArrowsNTp() / den);
							}

							den = (congI.getArrowsNTp() + congI.getArrowsNFn());
							if (den != 0.0){
								arrNRIc ++;
								arrNRI[s] += (congI.getArrowsNTp() / den);
							}

							den = (congI.getArrowsTp()+congI.getArrowsFp());
							if (den != 0.0){
								arrPI[s] += (congI.getArrowsTp() / den);
							}

							den = (congI.getArrowsTp()+congI.getArrowsFn());
							if (den != 0.0){
								arrRIc ++;
								arrRI[s] += (congI.getArrowsTp() / den);
							}

							den = (conAdjGI.getAdjITp() + conAdjGI.getAdjIFp());
							if (den != 0.0){
								adjIPI[s] += (conAdjGI.getAdjITp() / den);
							}

							den = (conAdjGI.getAdjNTp() + conAdjGI.getAdjNFp());
							if (den != 0.0){
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
							adjPI[s] += (conAdjGI.getAdjTp() / den);

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
								arrIP[s] += (cong.getArrowsITp() / den);
							}

							den = (cong.getArrowsITp() + cong.getArrowsIFn());
							if(den != 0.0){
								arrIRc ++;
								arrIR[s] += (cong.getArrowsITp() / den);
							}

							den = (cong.getArrowsNTp() + cong.getArrowsNFp());
							if (den != 0.0){
								arrNP[s] += (cong.getArrowsNTp() / den);
							}

							den = (cong.getArrowsNTp() + cong.getArrowsNFn());
							if (den != 0.0){
								arrNRc ++;
								arrNR[s] += (cong.getArrowsNTp() / den);
							}

							den = (cong.getArrowsTp() + cong.getArrowsFp());
							if (den != 0.0){
								arrP[s] += (cong.getArrowsTp() / den);
							}

							den = (cong.getArrowsTp() + cong.getArrowsFn());

							if (den != 0.0){
								arrRc ++;
								arrR[s] += (cong.getArrowsTp() / den);
							}

							den = (conAdjG.getAdjITp() + conAdjG.getAdjIFp());
							if (den != 0.0){
								adjIP[s] += (conAdjG.getAdjITp() / den);
							}

							den = (conAdjG.getAdjITp() + conAdjG.getAdjIFn());
							if(den != 0.0){
								adjIRc ++;
								adjIR[s] += (conAdjG.getAdjITp() / den);
							}

							den = (conAdjG.getAdjNTp() + conAdjG.getAdjNFp());
							if (den != 0.0){
								adjNP[s] += (conAdjG.getAdjNTp() / den);
							}
							den = (conAdjG.getAdjNTp() + conAdjG.getAdjNFn());
							if (den != 0.0){
								adjNRc ++;
								adjNR[s] += (conAdjG.getAdjNTp() / den);
							}

							den = (conAdjG.getAdjTp() + conAdjG.getAdjFp());
							if (den != 0.0){
								adjP[s] += (conAdjG.getAdjTp() / den);
							}

							den = (conAdjG.getAdjTp() + conAdjG.getAdjFn());
							if (den != 0.0){
								adjRc ++;
								adjR[s] += (conAdjG.getAdjTp() / den);
							}
							//				System.out.println("-------------");

							GraphUtils.GraphComparison cmpI = SearchGraphUtils.getGraphComparison(graphI, truePag);
							GraphUtils.GraphComparison cmpP = SearchGraphUtils.getGraphComparison(graphP, truePag);
							addedI[s] += cmpI.getEdgesAdded().size();
							removedI[s] += cmpI.getEdgesRemoved().size();
							reorientedI[s] += cmpI.getEdgesReorientedTo().size();

							added[s] += cmpP.getEdgesAdded().size();
							removed[s] += cmpP.getEdgesRemoved().size();
							reoriented[s] += cmpP.getEdgesReorientedTo().size();

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
						avgcsi[s] /= (numVars * numTests);
						System.out.println("avgsci : "+ avgcsi[s]);

						arrIPI[s] /= numTests;
						arrIRI[s] /= arrIRIc;
						arrNPI[s] /= numTests;
						arrNRI[s] /= arrNRIc;
						arrPI[s] /= numTests;
						arrRI[s] /= arrRIc;
						adjIPI[s] /= numTests;
						adjIRI[s] /= adjIRIc;
						adjNPI[s] /= numTests;
						adjNRI[s] /= adjNRIc;
						adjPI[s] /= numTests;
						adjRI[s] /= adjRIc;
						addedI[s] /= numTests;
						removedI[s] /= numTests;
						reorientedI[s] /= numTests;

						addedI_IS[s] /= numTests;
						removedI_IS[s] /= numTests;
						reorientedI_IS[s] /= numTests;
						addedI_Other[s] /= numTests;
						removedI_Other[s] /= numTests;
						reorientedI_Other[s] /= numTests;
						//		llrI[s] /= numTests;

						arrIP[s] /= numTests;
						arrIR[s] /= arrIRc;
						arrNP[s] /= numTests;
						arrNR[s] /= arrNRc;
						arrP[s] /= numTests;
						arrR[s] /= arrRc;
						adjIP[s] /= numTests;
						adjIR[s] /= adjIRc;
						adjNP[s] /= numTests;
						adjNR[s] /= adjNRc;
						adjP[s] /= numTests;
						adjR[s] /= adjRc;
						added[s] /= numTests;
						removed[s] /= numTests;
						reoriented[s] /= numTests;
						addedIS[s] /= numTests;
						removedIS[s] /= numTests;
						reorientedIS[s] /= numTests;
						addedOther[s] /= numTests;
						removedOther[s] /= numTests;
						reorientedOther[s] /= numTests;
						llr[s] /= numTests;


					}
					double[] pr_is = computePrecision(p_bsc_is, truth_bsc_is);
					double[] pr_pop = computePrecision(p_bsc_pop, truth_bsc_pop);

					System.out.println("precision bsc_is : " + pr_is[0]);
					System.out.println("recall    bsc_is : " + pr_is[1]);
					System.out.println("precision bsc_pop: " + pr_pop[0]);
					System.out.println("recall    bsc_pop: " + pr_pop[1]);
					
					System.out.println(avgInDeg2/numSim);
					//printRes("CSI", numSim, adjIPI, adjNPI, adjPI, adjIRI, adjNRI, adjRI, addedI, removedI, reorientedI, llr);
					printRes(this.out, "CSI", numSim, arrIPI, arrNPI, arrPI, arrIRI, arrNRI, arrRI, adjIPI, adjNPI, adjPI, adjIRI, adjNRI, adjRI, addedI, removedI, reorientedI, addedI_IS, removedI_IS, reorientedI_IS, addedI_Other, removedI_Other, reorientedI_Other, llr);

					//printRes("POP", numSim, adjIP, adjNP, adjP, adjIR, adjNR, adjR, added, removed, reoriented, llr);
					printRes(this.out,"POP", numSim, arrIP, arrNP, arrP, arrIR, arrNR, arrR, adjIP, adjNP, adjP, adjIR, adjNR, adjR, added, removed, reoriented, addedIS, removedIS, reorientedIS, addedOther, removedOther, reorientedOther, llr);
					this.out.close();
					System.out.println(avgOutDeg2/numSim);
					System.out.println(avgInDeg2/numSim);
					System.out.println("----------------------");

				}
			}
		}
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
	private void printRes(PrintStream out, String alg, int numSim, double[] arrIPI, double[] arrNPI, double[] arrPI, double[] arrIRI, double[] arrNRI, double[] arrRI, double[] adjIPI, double[] adjNPI, double[] adjPI, double[] adjIRI, double[] adjNRI, double[] adjRI, double[] addedI, double[] removedI, double[] reorientedI, double[] addedI_IS, double[] removedI_IS, double[] reorientedI_IS, double[] addedI_Other, double[] removedI_Other, double[] reorientedI_Other, double[] llrI){
		NumberFormat nf = new DecimalFormat("0.00");
		//			NumberFormat smallNf = new DecimalFormat("0.00E0");

		TextTable table = new TextTable(numSim+2, 8);
		table.setTabDelimited(true);
		String header = ", adj_P_IS, adj_P_NS, adj_P, adj_R_IS, adj_R_NS, adj_R, arr_P_IS, arr_P_NS, arr_P, arr_R_IS, arr_R_NS, arr_R, added_IS, added_NS, added, removed_IS, removed_NS, removed, reoriented_IS, reoriented_NS, reoriented, llr";
		table.setToken(0, 0, alg);
		table.setToken(0, 1, header);
		double arrIP = 0.0, arrNP = 0.0, arrP = 0.0, arrIR = 0.0, arrNR = 0.0, arrR = 0.0,
				adjIP = 0.0, adjNP = 0.0, adjP = 0.0, adjIR = 0.0, adjNR = 0.0, adjR = 0.0,
				added = 0.0, removed = 0.0, reoriented = 0.0,
				addedIS = 0.0, removedIS = 0.0, reorientedIS = 0.0,
				addedNS = 0.0, removedNS = 0.0, reorientedNS = 0.0, llr = 0.0;
		for (int i = 0; i < numSim; i++){
			String res = "," +nf.format(adjIPI[i])+","+nf.format(adjNPI[i])+","+nf.format(adjPI[i])+","+ nf.format(adjIRI[i])+
					","+nf.format(adjNRI[i])+","+nf.format(adjRI[i])+","+
					nf.format(arrIPI[i])+","+nf.format(arrNPI[i])+","+nf.format(arrPI[i])+","+ nf.format(arrIRI[i])+
					","+nf.format(arrNRI[i])+","+nf.format(arrRI[i])+","+
					nf.format(addedI_IS[i])+","+nf.format(addedI_Other[i])+","+nf.format(addedI[i])+","+
					nf.format(removedI_IS[i])+","+nf.format(removedI_Other[i])+","+nf.format(removedI[i])+","+
					nf.format(reorientedI_IS[i])+","+nf.format(reorientedI_Other[i])+","+nf.format(reorientedI[i])+","+ nf.format(llrI[i]);
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
			llr += llrI[i];
		}
		String res =  ","+nf.format(adjIP/numSim)+","+nf.format(adjNP/numSim)+","+nf.format(adjP/numSim)+","+nf.format(adjIR/numSim)+","+nf.format(adjNR/numSim)+","+nf.format(adjR/numSim)+","+
				nf.format(arrIP/numSim)+","+nf.format(arrNP/numSim)+","+nf.format(arrP/numSim)+","+nf.format(arrIR/numSim)+","+nf.format(arrNR/numSim)+","+nf.format(arrR/numSim)+","+
				nf.format(addedIS/numSim)+","+nf.format(addedNS/numSim)+","+nf.format(added/numSim)+","+
				nf.format(removedIS/numSim)+","+nf.format(removedNS/numSim)+","+nf.format(removed/numSim)+","+
				nf.format(reorientedIS/numSim)+","+nf.format(reorientedNS/numSim)+","+nf.format(reoriented/numSim)+","+nf.format(llr/numSim);
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
