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
import java.util.concurrent.ConcurrentMap;


import edu.cmu.tetrad.algcomparison.statistic.utils.AdjacencyConfusionIS;
import edu.cmu.tetrad.algcomparison.statistic.utils.ArrowConfusionIS;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.MlBayesIm;
import edu.cmu.tetrad.bayes.ISMlBayesIm;
import edu.cmu.tetrad.data.DataSet;
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

public class TestISFges {
	private ConcurrentMap<Node, Integer> hashIndices ;
	private PrintStream out;
	public static void main(String[] args) {
		
		TestISFges t = new TestISFges();
		t.testSimulation();
	}

	public void testSimulation(){
		RandomUtil.getInstance().setSeed(1454147770L);
		int[] numVarss = new int[]{50};
		double[] edgesPerNodes = new double[]{2.0};
		int numCases = 1000;
		int numTests = 500;
		int minCat = 2;
		int maxCat = 3;
		int numSim = 10;
		double k_add = 0.1;
		double k_delete = k_add; 
		double k_reverse = k_add; 
		for (int numVars: numVarss){
			for (double edgesPerNode : edgesPerNodes){
				final int numEdges = (int) (numVars * edgesPerNode);
				double avgInDeg2 = 0.0, avgOutDeg2 = 0.0;
				System.out.println("# nodes: " + numVars + ", # edges: "+ numEdges + ", # training: " + numCases + ", # test: "+ numTests);
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
				double samplePrior = 10.0;
				try {
		            File dir = new File("/Users/fattanehjabbari/CCD-Project/CS-BN/simulation-newprior/PESS"+samplePrior+"/");
		            
		            dir.mkdirs();
		            String outputFileName = "V"+numVars +"-E"+ edgesPerNode +"-kadd" + k_add+"-kdel" + k_delete+"-krev" + k_reverse+ "-PESS" + samplePrior+"-np.csv";
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
					Graph trueBN = GraphUtils.randomGraphRandomForwardEdges(vars, 0, numEdges, 30, 15, 15, false, true);
//					for (Node nod: trueBN.getNodes()){
//						if (trueBN.getIndegree(nod)>1)
//							avgInDeg2 = avgInDeg2 + 1;
//						if (trueBN.getOutdegree(nod)>1)
//							avgOutDeg2  = avgOutDeg2 +1;
//					}
					BayesPm pm = new BayesPm(trueBN, minCat, maxCat);
					ISMlBayesIm im = new ISMlBayesIm(pm, ISMlBayesIm.RANDOM);
//					System.out.println(im);
					
					// simulate train and test data from BN
					DataSet trainData = im.simulateData(numCases, false, tiers);
					DataSet testData = im.simulateData(numTests, false, tiers);

					// learn the population model
					BDeuScore scoreP = new BDeuScore(trainData);
					scoreP.setSamplePrior(samplePrior);
					Fges fgesP = new Fges (scoreP);
					//fgesP.setKnowledge(knowledge);
					Graph graphP = fgesP.search();

					// estimate MAP parameters from the population model
					DagInPatternIterator iterator = new DagInPatternIterator(graphP);
					Graph dagP = iterator.next();
//					BayesPm pmP = new BayesPm(dagP);
//					//			BayesPm pmP = new BayesPm(graphP);
//
//					DirichletBayesIm priorP = DirichletBayesIm.symmetricDirichletIm(pmP, 1.0);
//					BayesIm imP = DirichletEstimator.estimate(priorP, trainData);
					//			System.out.println("trueBN: " + trueBN);
					double arrIRc = 0.0, arrNRc = 0.0, arrRc = 0.0, arrIRIc = 0.0, arrNRIc = 0.0, arrRIc = 0.0;
					double adjIRc = 0.0, adjNRc = 0.0, adjRc = 0.0, adjIRIc = 0.0, adjNRIc = 0.0, adjRIc = 0.0;
					double csi = 0.0;
					for (int i = 0; i < testData.getNumRows(); i++){
						DataSet test = testData.subsetRows(new int[]{i});
						if (i%100 == 0) {System.out.println(i);}
//						System.out.println("test: " + test);

						// obtain the true instance-specific BN
						Map <Node, Boolean> context= new HashMap<Node, Boolean>();
						Graph trueBNI = SearchGraphUtils.patternForDag(new EdgeListGraph(GraphUtils.getISGraph(trueBN, im, test, context)));
//						System.out.println("context: " + context);
						
						for (Node n: context.keySet()){
							if (context.get(n)){
								avgcsi[s] += 1;
							}
						}
					
						// learn the instance-specific model
						ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
						scoreI.setKAddition(k_add);
						scoreI.setKDeletion(k_delete);
						scoreI.setKReorientation(k_reverse);
						scoreI.setSamplePrior(samplePrior);
						ISFges fgesI = new ISFges(scoreI);
						fgesI.setPopulationGraph(graphP);
						fgesI.setInitialGraph(graphP);
//						int [] parents = new int[2];
//						parents[0] = 0;
//						parents[1] = 3;
////						parents[2] = 9;
//						int [] parents_pop = new int[2];
//						parents_pop[0] = 0;
//						parents_pop[1] = 3;
////						parents_pop[2] = 4;
////						parents_pop[3] = 5;
//						int [] children_pop = new int[0];
//						scoreI.localScore(8, parents, parents_pop, children_pop);
						Graph graphI = fgesI.search();
					
	
						ArrowConfusionIS congI = new ArrowConfusionIS(trueBNI, GraphUtils.replaceNodes(graphI, trueBNI.getNodes()), context);
						AdjacencyConfusionIS conAdjGI = new AdjacencyConfusionIS(trueBNI, GraphUtils.replaceNodes(graphI, trueBNI.getNodes()), context);
						
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
						ArrowConfusionIS cong = new ArrowConfusionIS(trueBNI, GraphUtils.replaceNodes(graphP, trueBNI.getNodes()), context);
						AdjacencyConfusionIS conAdjG = new AdjacencyConfusionIS(trueBNI, GraphUtils.replaceNodes(graphP, trueBNI.getNodes()), context);

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

						GraphUtils.GraphComparison cmpI = SearchGraphUtils.getGraphComparison(graphI, trueBNI);
						GraphUtils.GraphComparison cmpP = SearchGraphUtils.getGraphComparison(graphP, trueBNI);
						addedI[s] += cmpI.getEdgesAdded().size();
						removedI[s] += cmpI.getEdgesRemoved().size();
						reorientedI[s] += cmpI.getEdgesReorientedTo().size();
						
						added[s] += cmpP.getEdgesAdded().size();
						removed[s] += cmpP.getEdgesRemoved().size();
						reoriented[s] += cmpP.getEdgesReorientedTo().size();

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
						
						//// learn a pop model from data + test
						//DataSet data = DataUtils.concatenate(trainData, test);
						DagInPatternIterator iteratorI = new DagInPatternIterator(graphI);
						Graph dagI = iteratorI.next();
//						BayesPm pmI = new BayesPm(dagI);
//						//				BayesPm pmI = new BayesPm(graphI);
//						//System.out.println("dagI: " + dagI);
//						DirichletBayesIm priorI = DirichletBayesIm.symmetricDirichletIm(pmI, 1.0);
//						BayesIm imI = DirichletEstimator.estimate(priorI, trainData);

						//llrI[s] += getLikelihood(imI, test) - getLikelihood(im, test);
						llr[s] += fgesI.scoreDag(dagI) - fgesI.scoreDag(dagP);
						
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
//						printRes("CSI", numSim, adjIPI, adjNPI, adjPI, adjIRI, adjNRI, adjRI, addedI, removedI, reorientedI, llr);
				printRes(this.out, "CSI", numSim, arrIPI, arrNPI, arrPI, arrIRI, arrNRI, arrRI, adjIPI, adjNPI, adjPI, adjIRI, adjNRI, adjRI, addedI, removedI, reorientedI, addedI_IS, removedI_IS, reorientedI_IS, addedI_Other, removedI_Other, reorientedI_Other, llr);

				//		printRes("POP", numSim, adjIP, adjNP, adjP, adjIR, adjNR, adjR, added, removed, reoriented, llr);
				printRes(this.out,"POP", numSim, arrIP, arrNP, arrP, arrIR, arrNR, arrR, adjIP, adjNP, adjP, adjIR, adjNR, adjR, added, removed, reoriented, addedIS, removedIS, reorientedIS, addedOther, removedOther, reorientedOther, llr);
				this.out.close();
				System.out.println(avgOutDeg2/numSim);
				System.out.println(avgInDeg2/numSim);
				System.out.println("----------------------");

			}
		}
	}

	public void test1(){
		int numVars = 4;
		int numCases = 2000;
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

		System.out.println("IM:" + im);
		DataSet data = im.simulateData(numCases, false);
		DataSet test = im.simulateData(1, false);
		test.setDouble(0, 0, 1);
		test.setDouble(0, 1, 1);
		test.setDouble(0, 2, 1);
		test.setDouble(0, 3, 1);
		
		BDeuScore popScore = new BDeuScore(data);
		popScore.setSamplePrior(1);
		Fges popFges = new Fges (popScore);
		Graph outP = popFges.search();
		ISBDeuScore csi = new ISBDeuScore(data, test);
		csi.setSamplePrior(1);
		ISFges fgs = new ISFges(csi);
		fgs.setPopulationGraph(SearchGraphUtils.chooseDagInPattern(outP));
		fgs.setInitialGraph(SearchGraphUtils.chooseDagInPattern(outP));
		fgs.setVerbose(true);
		Graph out = fgs.search();
		
		System.out.println("test: " +test);
		System.out.println("Dag: "+dag);
		System.out.println("Pop: " + outP);//SearchGraphUtils.chooseDagInPattern(outP));
		System.out.println("IS: " + out+"\n");//(SearchGraphUtils.chooseDagInPattern(out)));
		System.out.println("PS_score = " + fgs.scoreDag(SearchGraphUtils.chooseDagInPattern(out))+"\n");
		System.out.println("Pop_score = " + popFges.scoreDag(SearchGraphUtils.chooseDagInPattern(outP)));
		
	}
	
	public void test3(){
		int numCases = 300;
		int minCat = 2;
		int maxCat = 2;

		List<Node> vars = new ArrayList<>();
//		for (int i = 0; i < numVars; i++) {
		vars.add(new DiscreteVariable("Y"));
		vars.add(new DiscreteVariable("Z"));
		vars.add(new DiscreteVariable("X"));

//		}

		Graph dag = new EdgeListGraph(vars);
		dag.addDirectedEdge(dag.getNode("Y"), dag.getNode("X"));
		dag.addDirectedEdge(dag.getNode("Z"), dag.getNode("X"));
		
		BayesPm pm = new BayesPm(dag, minCat, maxCat);
		MlBayesIm im = new MlBayesIm(pm, MlBayesIm.MANUAL);
		im.setProbability(0, 0, 0, 0.75);
		im.setProbability(0, 0, 1, 0.25);
		im.setProbability(1, 0, 0, 0.51);
		im.setProbability(1, 0, 1, 0.49);
		im.setProbability(2, 0, 0, 0.9);
		im.setProbability(2, 0, 1, 0.1);
		im.setProbability(2, 1, 0, 0.9);
		im.setProbability(2, 1, 1, 0.1);
		im.setProbability(2, 2, 0, 0.23);
		im.setProbability(2, 2, 1, 0.77);
		im.setProbability(2, 3, 0, 0.52);
		im.setProbability(2, 3, 1, 0.48);
		

		System.out.println("IM:" + im);
		DataSet data = im.simulateData(numCases, false);
		DataSet test = im.simulateData(1, false);
		test.setDouble(0, 0, 0);
		test.setDouble(0, 1, 1);
		test.setDouble(0, 2, 1);
		
		BDeuScore popScore = new BDeuScore(data);
		Fges popFges = new Fges (popScore);
		Graph outP = popFges.search();

		ISBDeuScore csi = new ISBDeuScore(data, test);
		ISFges fgs = new ISFges(csi);
		fgs.setPopulationGraph(SearchGraphUtils.chooseDagInPattern(outP));
//		fgs.setInitialGraph(SearchGraphUtils.chooseDagInPattern(outP));
		Graph out = fgs.search();
		
		System.out.println("test: " +test);
		System.out.println("dag: "+dag);
		System.out.println("Pop: " + outP);
		System.out.println("IS: " + out + "\n");
		System.out.println("IS_score = " + fgs.scoreDag(SearchGraphUtils.chooseDagInPattern(out))+"\n");
		System.out.println("Pop_score = " + fgs.scoreDag(SearchGraphUtils.chooseDagInPattern(outP)));
		
	}
	private void printRes(PrintStream out, String alg, int numSim, double[] arrIPI, double[] arrNPI, double[] arrPI, double[] arrIRI, double[] arrNRI, double[] arrRI, double[] adjIPI, double[] adjNPI, double[] adjPI, double[] adjIRI, double[] adjNRI, double[] adjRI, double[] addedI, double[] removedI, double[] reorientedI, double[] addedI_IS, double[] removedI_IS, double[] reorientedI_IS, double[] addedI_Other, double[] removedI_Other, double[] reorientedI_Other, double[] llrI){
		NumberFormat nf = new DecimalFormat("0.00");
//		NumberFormat smallNf = new DecimalFormat("0.00E0");

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
//	private double getLikelihood(BayesIm im, DataSet dataSet) {
//
//		double lik = 0.0;
//
//		ROW:
//			for (int i = 0; i < dataSet.getNumRows(); i++) {
//				double lik0 = 0.0;
//
//				for (int j = 0; j < dataSet.getNumColumns(); j++) {
//					int[] parents = im.getParents(j);
//					int[] parentValues = new int[parents.length];
//
//					for (int k = 0; k < parents.length; k++) {
//						parentValues[k] = dataSet.getInt(i, parents[k]);
//					}
//
//					int dataValue = dataSet.getInt(i, j);
//					double p = im.getProbability(j, im.getRowIndex(j, parentValues), dataValue);
//
//					if (p == 0) continue ROW;
//
//					lik0 += Math.log(p);
//				}
//
//				lik += lik0;
//			}
//
//		return lik;
//	}


}
