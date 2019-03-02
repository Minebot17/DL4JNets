package org.deeplearning4j.rl4j;

import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteConv;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CThreadDiscrete;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraph;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdConv;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.factory.Nd4j;

public class TimeLimitLearning<O extends Encodable> extends A3CDiscreteConv<O> {

    public long limitTime;
    public long deltaTime;
    public long lastTime;

    public TimeLimitLearning(MDP<O, Integer, DiscreteSpace> mdp, IActorCritic IActorCritic, IHistoryProcessor.Configuration hpconf, A3CConfiguration conf, DataManager dataManager) {
        super(mdp, IActorCritic, hpconf, conf, dataManager);
    }

    public TimeLimitLearning(MDP<O, Integer, DiscreteSpace> mdp, ActorCriticFactoryCompGraph factory, IHistoryProcessor.Configuration hpconf, A3CConfiguration conf, DataManager dataManager) {
        super(mdp, factory.buildActorCritic(hpconf.getShape(), ((DiscreteSpace)mdp.getActionSpace()).getSize()), hpconf, conf, dataManager);
    }

    public TimeLimitLearning(MDP<O, Integer, DiscreteSpace> mdp, org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdConv.Configuration netConf, IHistoryProcessor.Configuration hpconf, A3CConfiguration conf, DataManager dataManager) {
        super(mdp, (ActorCriticFactoryCompGraph)(new ActorCriticFactoryCompGraphStdConv(netConf)), hpconf, conf, dataManager);
    }

    @Override
    public AsyncThread newThread(int i) {
        AsyncThread at = new TimeLimitThread(mdp.newInstance(), getAsyncGlobal(), getConfiguration(), i, getDataManager()).setLimit(limitTime);
        at.setHistoryProcessor(getHistoryProcessor());
        return at;
    }

    @Override
    public void launchThreads() {
        this.startGlobalThread();

        for(int i = 0; i < this.getConfiguration().getNumThread(); ++i) {
            Thread t = this.newThread(i);
            Nd4j.getAffinityManager().attachThreadToDevice(t, i % Nd4j.getAffinityManager().getNumberOfDevices());
            t.start();
        }
    }

    @Override
    public void train() {
        try {
            this.launchThreads();
            this.getDataManager().writeInfo(this);
            synchronized(this) {
                while(true) {
                    if (lastTime != 0)
                        deltaTime += System.currentTimeMillis() - lastTime;
                    lastTime = System.currentTimeMillis();
                    if (this.isTrainingComplete() || !this.getAsyncGlobal().isRunning() || limitTime <= deltaTime) {
                        break;
                    }

                    this.getPolicy().play(this.getMdp(), this.getHistoryProcessor());
                    this.getDataManager().writeInfo(this);
                    this.wait(20000L);
                }
            }
        } catch (Exception var4) {
            var4.printStackTrace();
        }

    }

    public TimeLimitLearning<O> setLimit(long milsecs){
        limitTime = milsecs;
        return this;
    }
}
