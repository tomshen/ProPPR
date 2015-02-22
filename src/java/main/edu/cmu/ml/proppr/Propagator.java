package edu.cmu.ml.proppr;

import edu.cmu.ml.proppr.examples.PosNegRWExample;
import edu.cmu.ml.proppr.graph.LearningGraph;
import edu.cmu.ml.proppr.graph.LearningGraphBuilder;
import edu.cmu.ml.proppr.graph.SimpleLearningGraph;
import edu.cmu.ml.proppr.learn.SRW;
import edu.cmu.ml.proppr.learn.tools.GroundedExampleStreamer;
import edu.cmu.ml.proppr.util.ParamVector;
import edu.cmu.ml.proppr.util.ParamsFile;
import edu.cmu.ml.proppr.util.ParsedFile;
import edu.cmu.ml.proppr.util.SimpleParamVector;
import edu.cmu.ml.proppr.util.multithreading.Multithreading;
import gnu.trove.list.TIntList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.procedure.TIntDoubleProcedure;
import org.apache.log4j.Logger;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Properties;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Created by tom on 2/19/15.
 */
public class Propagator {
    private static final Logger log = Logger.getLogger(Propagator.class);
    public static final int DEFAULT_CAPACITY = 16;
    public static final float DEFAULT_LOAD = (float) 0.75;

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


    public static void main(String[] args) {
        ParsedFile graphFile = new ParsedFile(args[0]);
        BufferedWriter resultsWriter = null;
        try {
            resultsWriter = Files.newBufferedWriter(FileSystems.getDefault().getPath(args[1]), Charset.defaultCharset());
        } catch (IOException e) {
            e.printStackTrace();
        }
        int nthreads = Integer.parseInt(args[2]);
        SRW srw = new edu.cmu.ml.proppr.learn.L2PosNegLossTrainedSRW();

        long start = System.currentTimeMillis();
        LearningGraphBuilder graphBuilder = new SimpleLearningGraph.SLGBuilder();

        while (graphFile.hasNext()) {
            // Parsing based on edu.cmu.ml.proppr.learn.tools.GroundedExampleStreamer
            String line = graphFile.next();
            log.debug("Importing example from line " + graphFile.getLineNumber());
            String[] parts = line.trim().split("\t", 5);
            String label = parts[0];

            int startNode = Integer.parseInt(parts[2].split(",")[0]);

            try {
                LearningGraph g = graphBuilder.deserialize(parts[4]);

                TIntDoubleMap startVec = new TIntDoubleHashMap();
                startVec.put(startNode, 1.0);

                ParamVector params = createParamVector(nthreads);
                srw.setupParams(params);

                TIntDoubleMap resultVec = srw.rwrUsingFeatures(g, startVec, params);
                if (resultsWriter != null) {
                    resultsWriter.write(label + "\n" + getTopResults(resultVec, 1e-6).toString() + "\n");
                }
            } catch (LearningGraph.GraphFormatException e) {
                log.error("Failed to deserialize learning graph.");
                e.printStackTrace();
                System.exit(-1);
            } catch (IOException e) {
                log.error("Failed to write results.");
                e.printStackTrace();
                System.exit(-1);
            }
        }
        log.info("Finished SRW in " + (System.currentTimeMillis() - start) + " ms");
    }
}
