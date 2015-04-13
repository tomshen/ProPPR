package edu.cmu.ml.proppr;

import edu.cmu.ml.proppr.graph.LearningGraph;
import edu.cmu.ml.proppr.graph.LearningGraphBuilder;
import edu.cmu.ml.proppr.graph.SimpleLearningGraph;
import edu.cmu.ml.proppr.learn.SRW;
import edu.cmu.ml.proppr.util.ParamVector;
import edu.cmu.ml.proppr.util.SRWOptions;
import edu.cmu.ml.proppr.util.SimpleParamVector;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import gnu.trove.procedure.TIntDoubleProcedure;
import gnu.trove.procedure.TIntObjectProcedure;
import org.apache.log4j.Logger;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.*;

/**
 * Created by tom on 2/19/15.
 */
public class Propagator {
    private static final Logger log = Logger.getLogger(Propagator.class);
    public static final int DEFAULT_CAPACITY = 16;
    public static final float DEFAULT_LOAD = (float) 0.75;
    public static final int iterations = 11;

    private static final int FEATURE_SEED = 1;
    private static final int FEATURE_ASSOC = 2;

    private static Map<String,String> parseJuntoConfig(String juntoConfigFile) {
        Map<String,String> config = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(juntoConfigFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] tokens = line.trim().split(" = ");
                if (tokens.length >= 2) {
                    config.put(tokens[0], tokens[1]);
                }
            }
        } catch (FileNotFoundException e) {
            log.trace("Junto config file not found: " + juntoConfigFile, e);
        } catch (IOException e) {
            log.trace("Error reading Junto config file: " + juntoConfigFile, e);
        }
        return config;
    }

    /**
     * Maps Strings to unique ints
     */
    private static class NodeMap {
        private TObjectIntMap<String> nodeMap;
        private TIntObjectMap<String> reverseNodeMap;
        private int nextInt;

        public NodeMap() {
            this.nodeMap = new TObjectIntHashMap<>();
            this.reverseNodeMap = new TIntObjectHashMap<>();
            this.nextInt = 2;
        }

        public boolean contains(String key) {
            return nodeMap.containsKey(key);
        }

        public int get(String key) {
            return nodeMap.get(key);
        }

        public String get(int key) {
            return reverseNodeMap.get(key);
        }

        public void put(String key) {
            nodeMap.putIfAbsent(key, nextInt);
            reverseNodeMap.putIfAbsent(nodeMap.get(key), key);
            nextInt++;
        }

        public int getStart() {
            return 1;
        }
    }

    private static String convertGraph(String graphFile, NodeMap nodeMap) {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(graphFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] tokens = line.trim().split("\t");
                if (tokens.length == 3) {
                    String node1 = tokens[0];
                    String node2 = tokens[1];
                    String weight = tokens[2];
                    nodeMap.put(node1);
                    nodeMap.put(node2);
                    sb.append(nodeMap.get(node1));
                    sb.append("->");
                    sb.append(nodeMap.get(node2));
                    sb.append(":");
                    sb.append(FEATURE_ASSOC);
                    sb.append("@");
                    sb.append(weight);
                    sb.append("\t");
                    sb.append(nodeMap.get(node2));
                    sb.append("->");
                    sb.append(nodeMap.get(node1));
                    sb.append(":");
                    sb.append(FEATURE_ASSOC);
                    sb.append("@");
                    sb.append(weight);
                    sb.append("\t");
                }
            }
        } catch (FileNotFoundException e) {
            log.error("Graph file not found: " + graphFile, e);
            System.exit(1);
        } catch (IOException e) {
            log.error("Error reading graph file: " + graphFile, e);
            System.exit(1);
        }
        return sb.toString();
    }

    private static Map<String,String> convertSeeds(String seedsFile, NodeMap nodeMap) {
        Map<String,StringBuilder> seedEdgeBuilder = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(seedsFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] tokens = line.trim().split("\t");
                if (tokens.length >= 2) {
                    String node = tokens[0];
                    String label = tokens[1];
                    if (!seedEdgeBuilder.containsKey(label)) {
                        seedEdgeBuilder.put(label, new StringBuilder());
                    }
                    StringBuilder sb = seedEdgeBuilder.get(label);
                    nodeMap.put(node);
                    sb.append(nodeMap.getStart());
                    sb.append("->");
                    sb.append(nodeMap.get(node));
                    sb.append(":");
                    sb.append(FEATURE_SEED);
                    sb.append("\t");
                }
            }
        } catch (FileNotFoundException e) {
            log.error("Seeds file not found: " + seedsFile, e);
        } catch (IOException e) {
            log.error("Error reading seeds file: " + seedsFile, e);
        }
        Map<String,String> seedEdges = new HashMap<>();
        for (Map.Entry<String,StringBuilder> e : seedEdgeBuilder.entrySet()) {
            seedEdges.put(e.getKey(), e.getValue().toString());
        }
        return seedEdges;
    }

    private static ParamVector createParamVector(int nthreads) {
        return new SimpleParamVector<String>(new ConcurrentHashMap<String,Double>(DEFAULT_CAPACITY,
                DEFAULT_LOAD, nthreads));
    }

    private static class SRWRunner implements Callable<SRWResult> {
        private final String graphEdges;
        private final String seedEdges;
        private final int startNode;
        private final String label;
        private final LearningGraphBuilder builder;
        private final SRW srw;

        public SRWRunner(String graphEdges, String seedEdges, int startNode, String label, LearningGraphBuilder builder,
                         SRW srw) {
            this.graphEdges = graphEdges;
            this.seedEdges = seedEdges;
            this.startNode = startNode;
            this.label = label;
            this.builder = builder;
            this.srw = srw;
        }

        @Override
        public SRWResult call() throws Exception {
            log.info("Building graph for label " + this.label);
            LearningGraph graph = this.builder.deserialize(" \t \t" + this.graphEdges + seedEdges);
            log.info("Running SRW on graph for label " + this.label);
            TIntDoubleMap startVec = new TIntDoubleHashMap();
            startVec.put(this.startNode, 1.0);
            ParamVector params = createParamVector(1);
            this.srw.setupParams(params);
            TIntDoubleMap resultVec = this.srw.rwrUsingFeatures(graph, startVec, params);
            return new SRWResult(resultVec, label);
        }
    }

    private static class SRWResult {
        private TIntDoubleMap resultVec;
        private String label;

        public SRWResult(TIntDoubleMap resultVec, String label) {
            this.resultVec = resultVec;
            this.label = label;
        }

        public TIntDoubleMap getResultVec() {
            return resultVec;
        }

        public String getLabel() {
            return label;
        }
    }

    private static void putResult(final TIntObjectMap<Map<String,Double>> map, final SRWResult res) {
        res.getResultVec().forEachEntry(new TIntDoubleProcedure() {
            @Override
            public boolean execute(int node, double weight) {
                map.putIfAbsent(node, new HashMap<String, Double>());
                map.get(node).put(res.getLabel(), weight);
                return true;
            }
        });
    }

    private static TIntObjectMap<Map<String,Double>> executeSRW(String graphEdges, Map<String,String> labelSeedEdges,
                                                                int startNode, LearningGraphBuilder builder, SRW srw,
                                                                int nthreads) {
        final TIntObjectMap<Map<String,Double>> nodeLabels = new TIntObjectHashMap<>();

        if (nthreads == 1) {
            for (Map.Entry<String,String> entry : labelSeedEdges.entrySet()) {
                try {
                    String label = entry.getKey();
                    String seedEdges = entry.getValue();
                    SRWResult res = new SRWRunner(graphEdges, seedEdges, startNode, label, builder, srw).call();
                    putResult(nodeLabels, res);
                } catch (Exception e) {
                    log.error("SRW execution failed", e);
                }
            }
        } else {
            ExecutorService pool = Executors.newFixedThreadPool(nthreads);

            Collection<Callable<SRWResult>> tasks = new ArrayList<>();

            for (Map.Entry<String,String> entry : labelSeedEdges.entrySet()) {
                String label = entry.getKey();
                String seedEdges = entry.getValue();
                tasks.add(new SRWRunner(graphEdges, seedEdges, startNode, label, builder, srw));
            }

            try {
                List<Future<SRWResult>> results = pool.invokeAll(tasks);
                for(Future<SRWResult> result : results) {
                    putResult(nodeLabels, result.get());
                }
            } catch (InterruptedException e) {
                log.error("SRW execution interrupted", e);
                pool.shutdownNow();
            } catch (ExecutionException e) {
                log.error("SRW execution failed", e);
                pool.shutdownNow();
            }
            pool.shutdownNow();
        }

        return nodeLabels;
    }

    public static void main(String[] args) throws IOException {
        final BufferedWriter resultsWriter = Files.newBufferedWriter(FileSystems.getDefault().getPath(args[1]),
                Charset.defaultCharset());
        int nthreads;
        if (args.length >= 3) {
            nthreads = Integer.parseInt(args[2]);
        } else {
            nthreads = 1;
        }

        Map<String,String> juntoConfig = parseJuntoConfig(args[0]);
        final NodeMap nodeMap = new NodeMap();
        String graphEdges = convertGraph(juntoConfig.get("graph_file"), nodeMap);
        log.info("Converted graph file");
        Map<String,String> seedEdges = convertSeeds(juntoConfig.get("seed_file"), nodeMap);
        log.info("Converted seeds file");
        LearningGraphBuilder graphBuilder = new SimpleLearningGraph.SLGBuilder();

        SRW srw = new SRW<>(new SRWOptions(iterations));

        long start = System.currentTimeMillis();
        final TIntObjectMap<Map<String,Double>> nodeLabels = executeSRW(graphEdges, seedEdges, nodeMap.getStart(),
                graphBuilder, srw, nthreads);
        log.info("Finished SRW in " + (System.currentTimeMillis() - start) + " ms");

        start = System.currentTimeMillis();
        nodeLabels.forEachEntry(new TIntObjectProcedure<Map<String, Double>>() {
            private <K, V extends Comparable<? super V>> List<K>
            sortedKeysByValue(Map<K, V> map) {
                List<Map.Entry<K, V>> list =
                        new LinkedList<>(map.entrySet());
                Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
                    @Override
                    public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
                        return (o1.getValue()).compareTo(o2.getValue());
                    }
                });
                List<K> keys = new LinkedList<>();
                for (Map.Entry<K, V> entry : list) {
                    keys.add(entry.getKey());
                }
                return keys;
            }

            @Override
            public boolean execute(int node, Map<String, Double> labelWeights) {
                List<String> sortedLabels = sortedKeysByValue(labelWeights);
                Collections.reverse(sortedLabels);
                try {
                    StringBuilder sb = new StringBuilder();
                    for (String label : sortedLabels) {
                        sb.append(label + "\t");
                    }
                    resultsWriter.write(nodeMap.get(node) + "\t" + sb.toString() + "\n");
                } catch (IOException e) {
                    log.error("Failed to write node ", e);
                }
                return true;
            }
        });
        log.info("Finished writing results in " + (System.currentTimeMillis() - start) + " ms");
    }
}
