package edu.cmu.ml.proppr.learn.tools;

import edu.cmu.ml.proppr.graph.LearningGraph;
import edu.cmu.ml.proppr.graph.LearningGraphBuilder;
import edu.cmu.ml.proppr.util.FileBackedIterable;
import org.apache.log4j.Logger;

import java.util.Iterator;
import java.util.Map;

/**
 * Created by tom on 3/27/15.
 */
public class JuntoGraphStreamer implements Iterable<JuntoGraph>, Iterator<JuntoGraph>, FileBackedIterable {
    private static final Logger log = Logger.getLogger(JuntoGraphStreamer.class);
    public static final String MAJOR_DELIM="\t";
    public static final String MINOR_DELIM=",";
    private final String graphEdges;
    private final Map<String, String> labelSeedEdges;
    private Iterator<Map.Entry<String, String>> labelIterator;
    private final int startNode;
    private LearningGraphBuilder builder;
    public JuntoGraphStreamer(String graphEdges, Map<String,String> labelSeedEdges, int startNode,
                              LearningGraphBuilder builder) {
        this.graphEdges = graphEdges;
        this.labelSeedEdges = labelSeedEdges;
        this.labelIterator = this.labelSeedEdges.entrySet().iterator();
        this.startNode = startNode;
        this.builder = builder;
    }

    @Override
    public boolean hasNext() {
        return this.labelIterator.hasNext();
    }

    @Override
    public JuntoGraph next() {
        Map.Entry<String,String> entry = this.labelIterator.next();
        String label = entry.getKey();
        String seedEdges = entry.getValue();
        log.debug("Processing label " + label);
        try {
            LearningGraph g = this.builder.deserialize(" \t \t" + this.graphEdges + seedEdges);
            return new JuntoGraph(g, label, this.startNode);
        } catch (LearningGraph.GraphFormatException e) {
            if (this.hasNext()) return next();
            else return null;
        }
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Can't remove from a grounded example stream");
    }

    @Override
    public Iterator<JuntoGraph> iterator() {
        return this;
    }


    @Override
    public void wrap() {
        if (this.hasNext()) return;
        this.labelIterator = this.labelSeedEdges.entrySet().iterator();
    }

}