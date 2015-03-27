package edu.cmu.ml.proppr.learn.tools;

import edu.cmu.ml.proppr.graph.LearningGraph;

/**
 * Created by tom on 3/27/15.
 */
public class JuntoGraph {
    private LearningGraph g;
    private String label;
    private int startNode;

    public JuntoGraph(LearningGraph g, String label, int startNode) {
        this.g = g;
        this.label = label;
        this.startNode = startNode;
    }

    public LearningGraph getG() {
        return g;
    }

    public String getLabel() {
        return label;
    }

    public int getStartNode() {
        return startNode;
    }
}
