package edu.cmu.ml.proppr.learn.tools;

import edu.cmu.ml.proppr.examples.PosNegRWExample;

public class JuntoExample {
    public final String label;
    public PosNegRWExample example;

    public JuntoExample(String label, PosNegRWExample example) {
        this.label = label;
        this.example = example;
    }
}