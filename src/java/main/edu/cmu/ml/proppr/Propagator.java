package edu.cmu.ml.proppr;

import edu.cmu.ml.proppr.graph.LearningGraphBuilder;
import edu.cmu.ml.proppr.graph.SimpleLearningGraph;
import edu.cmu.ml.proppr.learn.SRW;
import edu.cmu.ml.proppr.learn.tools.JuntoGraph;
import edu.cmu.ml.proppr.learn.tools.JuntoGraphStreamer;
import edu.cmu.ml.proppr.util.ParamVector;
import edu.cmu.ml.proppr.util.ParsedFile;
import edu.cmu.ml.proppr.util.SRWOptions;
import edu.cmu.ml.proppr.util.SimpleParamVector;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.procedure.TIntDoubleProcedure;
import gnu.trove.procedure.TIntObjectProcedure;
import org.apache.log4j.Logger;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

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

    public static void main(String[] args) throws IOException {
        ParsedFile graphFile = new ParsedFile(args[0]);
        final BufferedWriter resultsWriter = Files.newBufferedWriter(FileSystems.getDefault().getPath(args[1]),
                Charset.defaultCharset());
        int nthreads = Integer.parseInt(args[2]);
        SRW srw = new SRW<>(new SRWOptions(iterations));

        long start = System.currentTimeMillis();
        LearningGraphBuilder graphBuilder = new SimpleLearningGraph.SLGBuilder();

        final TIntObjectMap<Map<String,Double>> nodeLabels = new TIntObjectHashMap<>();

        JuntoGraphStreamer graphs = new JuntoGraphStreamer(graphFile, new SimpleLearningGraph.SLGBuilder());

        for (final JuntoGraph graph : graphs) {
            TIntDoubleMap startVec = new TIntDoubleHashMap();
            startVec.put(graph.getStartNode(), 1.0);

            ParamVector params = createParamVector(nthreads);
            srw.setupParams(params);

            TIntDoubleMap resultVec = srw.rwrUsingFeatures(graph.getG(), startVec, params);
            resultVec.forEachEntry(new TIntDoubleProcedure() {
                @Override
                public boolean execute(int node, double weight) {
                    nodeLabels.putIfAbsent(node, new HashMap<String, Double>());
                    nodeLabels.get(node).put(graph.getLabel(), weight);
                    return true;
                }
            });
        }
        log.info("Finished SRW in " + (System.currentTimeMillis() - start) + " ms");

        nodeLabels.forEachEntry(new TIntObjectProcedure<Map<String, Double>>() {
            private <K, V extends Comparable<? super V>> List<K>
            sortedKeysByValue( Map<K, V> map )
            {
                List<Map.Entry<K, V>> list =
                        new LinkedList<>( map.entrySet() );
                Collections.sort( list, new Comparator<Map.Entry<K, V>>()
                {
                    @Override
                    public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 )
                    {
                        return (o1.getValue()).compareTo( o2.getValue() );
                    }
                } );
                List<K> keys = new LinkedList<>();
                for (Map.Entry<K, V> entry : list)
                {
                    keys.add(entry.getKey());
                }
                return keys;
            }
            @Override
            public boolean execute(int node, Map<String, Double> labelWeights) {
                List<String> sortedLabels = sortedKeysByValue(labelWeights);
                Collections.reverse(sortedLabels);
                if (resultsWriter != null) {
                    try {
                        StringBuilder sb = new StringBuilder();
                        for (String label : sortedLabels) {
                            sb.append(label + "\t");
                        }
                        resultsWriter.write(Integer.toString(node) + "\t" + sb.toString() + "\n");
                    } catch (IOException e) {
                        log.trace("Failed to write node ", e);
                    }
                }
                return true;
            }
        });
    }
}
