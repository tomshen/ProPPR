package edu.cmu.ml.proppr;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;

import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.procedure.TIntObjectProcedure;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;
import org.apache.log4j.Logger;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.procedure.TIntDoubleProcedure;

import edu.cmu.ml.proppr.graph.LearningGraph;
import edu.cmu.ml.proppr.graph.LearningGraphBuilder;
import edu.cmu.ml.proppr.graph.SimpleLearningGraph;
import edu.cmu.ml.proppr.learn.SRW;
import edu.cmu.ml.proppr.util.ParamVector;
import edu.cmu.ml.proppr.util.ParsedFile;
import edu.cmu.ml.proppr.util.SRWOptions;
import edu.cmu.ml.proppr.util.SimpleParamVector;

/**
 * Created by tom on 2/19/15.
 */
public class Propagator {
    private static final Logger log = Logger.getLogger(Propagator.class);
    public static final int DEFAULT_CAPACITY = 16;
    public static final float DEFAULT_LOAD = (float) 0.75;
    public static final int iterations = 11;

    private static ParamVector createParamVector(int nthreads) {
        return new SimpleParamVector<String>(new ConcurrentHashMap<String,Double>(DEFAULT_CAPACITY,
                DEFAULT_LOAD, nthreads));
    }

    private static SortedSet<Integer> getTopResults(TIntDoubleMap results, final double alpha) {
        final SortedSet<Integer> topResults = new TreeSet<>();
        results.forEachEntry(new TIntDoubleProcedure() {
            @Override
            public boolean execute(int i, double v) {
                if (v >= alpha) {
                    topResults.add(i);
                }
                return true;
            }
        });
        return topResults;
    }


    public static void main(String[] args) throws IOException {
        ParsedFile graphFile = new ParsedFile(args[0]);
        final BufferedWriter resultsWriter = Files.newBufferedWriter(FileSystems.getDefault().getPath(args[1]),
                Charset.defaultCharset());
        int nthreads = Integer.parseInt(args[2]);
        SRW srw = new SRW<>(new SRWOptions(iterations));

        long start = System.currentTimeMillis();
        LearningGraphBuilder graphBuilder = new SimpleLearningGraph.SLGBuilder();

        final TIntObjectMap<Map<String,Double>> nodeLabels = new TIntObjectHashMap<>();

        while (graphFile.hasNext()) {
            // Parsing based on edu.cmu.ml.proppr.learn.tools.GroundedExampleStreamer
            String line = graphFile.next();
            log.debug("Importing example from line " + graphFile.getLineNumber());
            String[] parts = line.trim().split("\t", 5);
            final String label = parts[0];

            int startNode = Integer.parseInt(parts[2].split(",")[0]);

            try {
                LearningGraph g = graphBuilder.deserialize(parts[4]);

                TIntDoubleMap startVec = new TIntDoubleHashMap();
                startVec.put(startNode, 1.0);

                ParamVector params = createParamVector(nthreads);
                srw.setupParams(params);

                TIntDoubleMap resultVec = srw.rwrUsingFeatures(g, startVec, params);
                resultVec.forEachEntry(new TIntDoubleProcedure() {
                    @Override
                    public boolean execute(int node, double weight) {
                        nodeLabels.putIfAbsent(node, new HashMap<String, Double>());
                        nodeLabels.get(node).put(label, weight);
                        return true;
                    }
                });
            } catch (LearningGraph.GraphFormatException e) {
                log.error("Failed to deserialize learning graph.");
                e.printStackTrace();
                System.exit(-1);
            }
        }
        log.info("Finished SRW in " + (System.currentTimeMillis() - start) + " ms");

        nodeLabels.forEachEntry(new TIntObjectProcedure<Map<String, Double>>() {
            private String argmax(Map<String, Double> stringDoubleMap) {
                double maxWeight = Double.MIN_VALUE;
                String maxWeightLabel = null;
                for (Map.Entry<String, Double> entry : stringDoubleMap.entrySet()) {
                    String label = entry.getKey();
                    double weight = entry.getValue();
                    if (weight > maxWeight) {
                        maxWeight = weight;
                        maxWeightLabel = label;
                    }
                }
                return maxWeightLabel;
            }
            @Override
            public boolean execute(int node, Map<String, Double> labelWeights) {
                String label = argmax(labelWeights);
                if (resultsWriter != null) {
                    try {
                        resultsWriter.write(Integer.toString(node) + "\t" + label + "\n");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                return true;
            }
        });
    }
}
