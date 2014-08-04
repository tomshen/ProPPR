package edu.cmu.ml.praprolog;

import edu.cmu.ml.praprolog.learn.SRW;
import edu.cmu.ml.praprolog.learn.tools.PosNegRWExample;

public class MultiRRMinibatchTrainer<T> extends MultithreadedRRTrainer<T> {

	public MultiRRMinibatchTrainer(SRW<PosNegRWExample<T>> learner,
			int numThreads) {
		super(learner, numThreads);
	}

	@Override
	protected void addThreads() {
		int k=0;
		for (TrainerExample ex : currentTrainingRun.queue) {
			currentTrainingRun.futures.add(currentTrainingRun.executor.submit(new TrainerThread(ex,k)));
			k++;
		}
	}
	
}
