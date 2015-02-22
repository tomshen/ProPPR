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
import edu.cmu.ml.proppr.util.multithreading.Multithreading;
import org.apache.log4j.Logger;

import java.io.File;

/**
 * Created by tom on 2/19/15.
 */
public class Propagator {
    private static final Logger log = Logger.getLogger(Propagator.class);
    public static void main(String[] args) {
        ParsedFile graphFile = new ParsedFile(args[0]);
        File paramsFile = new File(args[1]);
        SRW srw = new edu.cmu.ml.proppr.learn.L2PosNegLossTrainedSRW();
        Trainer trainer = new Trainer(srw, 8, Multithreading.DEFAULT_THROTTLE);
        long start = System.currentTimeMillis();
        ParamVector params = trainer.train(new GroundedExampleStreamer(graphFile,
                new SimpleLearningGraph.SLGBuilder()), 5, false);
        log.info("Finished SRW in " + (System.currentTimeMillis() - start) + " ms");
        if (paramsFile != null) {
            log.info("Saving parameters to " + paramsFile);
            ParamsFile.save(params, paramsFile, null);
        }
    }
}
