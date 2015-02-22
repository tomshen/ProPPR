package edu.cmu.ml.proppr;

import edu.cmu.ml.proppr.examples.PosNegRWExample;
import edu.cmu.ml.proppr.graph.SimpleLearningGraph;
import edu.cmu.ml.proppr.learn.SRW;
import edu.cmu.ml.proppr.learn.tools.JuntoExample;
import edu.cmu.ml.proppr.learn.tools.JuntoExampleStreamer;
import edu.cmu.ml.proppr.learn.tools.LossData;
import edu.cmu.ml.proppr.util.*;
import edu.cmu.ml.proppr.util.multithreading.Cleanup;
import edu.cmu.ml.proppr.util.multithreading.Multithreading;
import edu.cmu.ml.proppr.util.multithreading.Transformer;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.set.TIntSet;
import org.apache.log4j.Logger;

import java.io.*;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

/**
 * Created by tom on 2/19/15.
 */
public class Propagator {
    private static final Logger log = Logger.getLogger(Propagator.class);
    public static final int DEFAULT_CAPACITY = 16;
    public static final float DEFAULT_LOAD = (float) 0.75;
    protected SRW<PosNegRWExample> learner;
    protected int nthreads = 1;
    protected int throttle;
    protected int epoch;
    LossData lossLastEpoch;

    public Propagator(SRW<PosNegRWExample> learner, int nthreads, int throttle) {
        this.learner = learner;
        this.nthreads = Math.max(1, nthreads);
        this.throttle = throttle;

        learner.untrainedFeatures().add("fixedWeight");
        learner.untrainedFeatures().add("id(trueLoop)");
        learner.untrainedFeatures().add("id(trueLoopRestart)");
        learner.untrainedFeatures().add("id(restart)");
        learner.untrainedFeatures().add("id(alphaBooster)");
    }

    public Propagator(SRW<PosNegRWExample> learner) {
        this(learner, 22, Multithreading.DEFAULT_THROTTLE);
    }

    private ParamVector createParamVector() {
        return new SimpleParamVector<>(new ConcurrentHashMap<String,Double>(DEFAULT_CAPACITY,DEFAULT_LOAD,this.nthreads));
    }

    public Map<String, TIntSet> walk(Iterable<JuntoExample> examples, int numEpochs, boolean traceLosses) {
        return walk(
                examples,
                createParamVector(),
                numEpochs,
                traceLosses
        );
    }

    public Map<String, TIntSet> walk(Iterable<JuntoExample> examples, ParamVector initialParamVec, int numEpochs, boolean traceLosses) {
        ParamVector paramVec = this.learner.setupParams(initialParamVec);
        Map<String, TIntSet> labeledNodes = new HashMap<>();
        if (paramVec.size() == 0)
            for (String f : this.learner.untrainedFeatures()) paramVec.put(f, this.learner.getWeightingScheme().defaultWeight());
        Multithreading<TrainingExample,Integer> m = new Multithreading<TrainingExample,Integer>(log);
        // loop over epochs
        for (int i=0; i<numEpochs; i++) {
            // set up current epoch
            this.epoch++;
            this.learner.setEpoch(epoch);
            log.info("epoch "+epoch+" ...");

            this.learner.clearLoss();
            numExamplesThisEpoch = 0;

            if (examples instanceof FileBackedIterable) ((FileBackedIterable) examples).wrap();

            // run examples through Multithreading
            m.executeJob(
                    this.nthreads,
                    new TrainingExampleStreamer(examples, paramVec, labeledNodes),
                    new Transformer<TrainingExample,Integer>() {
                        @Override
                        public Callable<Integer> transformer(
                                TrainingExample in, int id) {
                            return new Train(in,id);
                        }
                    },
                    new Cleanup<Integer>() {
                        @Override
                        public Runnable cleanup(Future<Integer> in, int id) {
                            return new TraceLosses(in,id);
                        }
                    },
                    this.throttle);

            //
            this.learner.cleanupParams(paramVec);

            if(traceLosses) {
                LossData lossThisEpoch = this.learner.cumulativeLoss();
                for(Map.Entry<LossData.LOSS,Double> e : lossThisEpoch.loss.entrySet()) e.setValue(e.getValue() / numExamplesThisEpoch);
                System.out.print("avg training loss " + lossThisEpoch.total()
                        + " on "+ numExamplesThisEpoch +" examples");
                System.out.print(" =log:reg " + lossThisEpoch.loss.get(LossData.LOSS.LOG));
                System.out.print(" : " + lossThisEpoch.loss.get(LossData.LOSS.REGULARIZATION));
                if (epoch>1) {
                    LossData diff = lossLastEpoch.diff(lossThisEpoch);
                    System.out.println(" improved by " + diff.total()
                            + " (log:reg "+diff.loss.get(LossData.LOSS.LOG) +":"+diff.loss.get(LossData.LOSS.REGULARIZATION)+")");
                    if (diff.total() < 0.0) {
                        System.out.println("WARNING: loss INCREASED by " +
                                (diff.total()) + " - what's THAT about?");
                    }
                } else
                    System.out.println();

                lossLastEpoch = lossThisEpoch;

            }
        }
        return paramVec;
    }

    /////////////////////// Multithreading scaffold ///////////////////////

    /**
     * Stream over instances of this class
     * @author "Kathryn Mazaitis <krivard@cs.cmu.edu>"
     *
     */
    private class TrainingExample {
        PosNegRWExample x;
        ParamVector paramVec;
        String label;
        Map<String, TIntSet> labeledNodes;

        SRW learner;
        public TrainingExample(PosNegRWExample x, ParamVector paramVec, String label, Map<String, TIntSet> labeledNodes, SRW learner) {
            this.x = x;
            this.paramVec = paramVec;
            this.learner = learner;
            this.label = label;
            this.labeledNodes = labeledNodes;
        }
    }

    /**
     * Transforms from inputs to outputs
     * @author "Kathryn Mazaitis <krivard@cs.cmu.edu>"
     *
     */
    private class Train implements Callable<Integer> {
        TrainingExample in;
        int id;
        public Train(TrainingExample in, int id) {
            this.in = in;
            this.id = id;
        }
        @Override
        public Integer call() throws Exception {
            log.debug("Training on example "+this.id);
            in.learner.trainOnExample(in.paramVec, in.x);
            in.labeledNodes.put(in.label, in.learner.)
            return in.x.length();
        }
    }

    protected int numExamplesThisEpoch;
    /**
     * Cleans up outputs from training (tracks some info for traceLosses)
     * @author "Kathryn Mazaitis <krivard@cs.cmu.edu>"
     *
     */
    private class TraceLosses implements Runnable {
        Future<Integer> in;
        int id;
        public TraceLosses(Future<Integer> in, int id) {
            this.in = in;
            this.id = id;
        }
        @Override
        public void run() {
            try {
                log.debug("Cleaning up example "+this.id);
                numExamplesThisEpoch += this.in.get();
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (ExecutionException e) {
                log.error("Trouble with #"+id,e);
            }
        }
    }
    /**
     * Builds the streamer of all training inputs from the streamer of training examples.
     * @author "Kathryn Mazaitis <krivard@cs.cmu.edu>"
     *
     */
    private class TrainingExampleStreamer implements Iterable<TrainingExample>,Iterator<TrainingExample> {
        Iterator<JuntoExample> examples;
        ParamVector paramVec;
        Map<String, TIntSet> labeledNodes;
        public TrainingExampleStreamer(Iterable<JuntoExample> examples, ParamVector paramVec, Map<String, TIntSet> labeledNodes) {
            this.examples = examples.iterator();
            this.paramVec = paramVec;
            this.labeledNodes = labeledNodes;
        }
        @Override
        public Iterator<TrainingExample> iterator() {
            return this;
        }

        @Override
        public boolean hasNext() {
            return examples.hasNext();
        }

        @Override
        public TrainingExample next() {
            JuntoExample juntoExample = examples.next();
            return new TrainingExample(juntoExample.example, paramVec, juntoExample.label, labeledNodes, learner);
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("No removal of examples permitted during training!");
        }
    }

    public static void main(String[] args) {
        ParsedFile graphFile = new ParsedFile(args[0]);
        File outFile = new File(args[1]);
        SRW srw = new edu.cmu.ml.proppr.learn.L2PosNegLossTrainedSRW();
        Propagator propagator = new Propagator(srw);

        long start = System.currentTimeMillis();
        Map<String, TIntSet> labeledNodes = propagator.walk(new JuntoExampleStreamer(graphFile,
                new SimpleLearningGraph.SLGBuilder()), 5, false);
        log.info("Finished SRW in " + (System.currentTimeMillis() - start) + " ms");
        if (outFile != null) {
            log.info("Saving labeled nodes to " + outFile);
            Properties properties = new Properties();
            for (Map.Entry<String, TIntSet> entry : labeledNodes.entrySet()) {
                properties.put(entry.getKey(), entry.getValue().toString());
            }
            try {
                properties.store(new FileOutputStream(outFile), null);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
