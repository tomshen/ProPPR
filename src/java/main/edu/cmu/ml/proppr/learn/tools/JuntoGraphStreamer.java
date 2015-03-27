package edu.cmu.ml.proppr.learn.tools;

import edu.cmu.ml.proppr.graph.LearningGraph;
import edu.cmu.ml.proppr.graph.LearningGraphBuilder;
import edu.cmu.ml.proppr.util.FileBackedIterable;
import edu.cmu.ml.proppr.util.ParsedFile;
import org.apache.log4j.Logger;

import java.util.Iterator;

/**
 * Created by tom on 3/27/15.
 */
public class JuntoGraphStreamer implements Iterable<JuntoGraph>, Iterator<JuntoGraph>, FileBackedIterable {
    private static final Logger log = Logger.getLogger(JuntoGraphStreamer.class);
    public static final String MAJOR_DELIM="\t";
    public static final String MINOR_DELIM=",";
    private ParsedFile file;
    private LearningGraphBuilder builder;
    public JuntoGraphStreamer(String cookedExamplesFile, LearningGraphBuilder builder) {
        this(new ParsedFile(cookedExamplesFile), builder);
    }
    public JuntoGraphStreamer(ParsedFile cookedExamplesFile, LearningGraphBuilder builder) {
        log.info("Importing cooked examples from "+cookedExamplesFile.getFileName());
        this.file = cookedExamplesFile;
        this.builder = builder;
    }

    @Override
    public boolean hasNext() {
        return this.file.hasNext();
    }

    private int[] stringToInt(String[] raw) {
        int[] ret = new int[raw.length];
        for (int i=0; i<raw.length; i++) { ret[i] = Integer.parseInt(raw[i]); }
        return ret;
    }

    @Override
    public JuntoGraph next() {
        String line = this.file.next();
        log.debug("Importing example from line "+file.getLineNumber());

        String[] parts = line.trim().split(MAJOR_DELIM,5);

        int[] posList;
        if (parts[2].length()>0) posList = stringToInt(parts[2].split(MINOR_DELIM));
        else posList = new int[0];
        String label = parts[0];
        int startNode = posList[0];
        try {
            LearningGraph g = builder.deserialize(parts[4]);
            return new JuntoGraph(g, label, startNode);
        } catch (LearningGraph.GraphFormatException e) {
            file.parseError("["+e.getMessage()+"]");
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
        this.file.reset();
    }

}