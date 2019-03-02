package org.deeplearning4j.rl4j;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.learning.async.MiniTrans;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CThreadDiscrete;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Stack;

public class TimeLimitThread<O extends Encodable> extends A3CThreadDiscrete<O> {

    public long limitTime;
    public long deltaTime;
    public long lastTime;

    public TimeLimitThread(MDP mdp, AsyncGlobal asyncGlobal, A3CDiscrete.A3CConfiguration a3cc, int threadNumber, DataManager dataManager) {
        super(mdp, asyncGlobal, a3cc, threadNumber, dataManager);
    }

    public TimeLimitThread setLimit(long milSecs){
        limitTime = milSecs;
        return this;
    }

    @Override
    public SubEpochReturn<O> trainSubEpoch(O sObs, int nstep) {
        synchronized(this.getAsyncGlobal()) {
            getCurrent().copy(this.getAsyncGlobal().getCurrent());
        }

        Stack<MiniTrans<Integer>> rewards = new Stack();
        O obs = sObs;
        Policy<O, Integer> policy = this.getPolicy(getCurrent());
        Integer lastAction = null;
        IHistoryProcessor hp = this.getHistoryProcessor();
        int skipFrame = hp != null ? hp.getConf().getSkipFrame() : 1;
        double reward = 0.0D;
        double accuReward = 0.0D;

        Integer action;
        int i;
        INDArray input;
        INDArray hstack;
        StepReply stepReply;
        for(i = 0; !this.getMdp().isDone() && i < nstep * skipFrame; lastAction = action) {
            if (lastTime != 0)
                deltaTime += System.currentTimeMillis() - lastTime;
            lastTime = System.currentTimeMillis();
            if (limitTime <= deltaTime)
                break;

            input = Learning.getInput(this.getMdp(), obs);
            hstack = null;
            if (hp != null) {
                hp.record(input);
            }

            if (i % skipFrame != 0 && lastAction != null) {
                action = lastAction;
            } else {
                hstack = this.processHistory(input);
                action = (Integer)policy.nextAction(hstack);
            }

            stepReply = this.getMdp().step(action);
            accuReward += stepReply.getReward() * this.getConf().getRewardFactor();
            if (i % skipFrame == 0 || lastAction == null || stepReply.isDone()) {
                obs = (O) stepReply.getObservation();
                if (hstack == null) {
                    hstack = this.processHistory(input);
                }

                INDArray[] output = getCurrent().outputAll(hstack);
                rewards.add(new MiniTrans(hstack, action, output, accuReward));
                accuReward = 0.0D;
            }

            reward += stepReply.getReward();
            ++i;
        }

        input = Learning.getInput(this.getMdp(), obs);
        hstack = this.processHistory(input);
        if (hp != null) {
            hp.record(input);
        }

        if (this.getMdp().isDone() && i < nstep * skipFrame) {
            rewards.add(new MiniTrans(hstack, (Object)null, (INDArray[])null, 0.0D));
        } else {
            stepReply = null;
            INDArray[] output;
            if (this.getConf().getTargetDqnUpdateFreq() == -1) {
                output = getCurrent().outputAll(hstack);
            } else {
                synchronized(this.getAsyncGlobal()) {
                    output = this.getAsyncGlobal().getTarget().outputAll(hstack);
                }
            }

            double maxQ = Nd4j.max(output[0]).getDouble(0L);
            rewards.add(new MiniTrans(hstack, (Object)null, output, maxQ));
        }

        this.getAsyncGlobal().enqueue(this.calcGradient(getCurrent(), rewards), i);
        return new SubEpochReturn(i, obs, reward, getCurrent().getLatestScore());
    }

    @Override
    public void run() {
        try {
            this.getCurrent().reset();
            Learning.InitMdp<O> initMdp = Learning.initMdp(this.getMdp(), getHistoryProcessor());
            O obs = initMdp.getLastObs();
            double rewards = initMdp.getReward();
            int length = initMdp.getSteps();
            this.preEpoch();

            while(!this.getAsyncGlobal().isTrainingComplete() && this.getAsyncGlobal().isRunning() && limitTime > deltaTime) {
                if (lastTime != 0)
                    deltaTime += System.currentTimeMillis() - lastTime;
                lastTime = System.currentTimeMillis();

                int maxSteps = Math.min(this.getConf().getNstep(), this.getConf().getMaxEpochStep() - length);
                AsyncThread.SubEpochReturn<O> subEpochReturn = this.trainSubEpoch(obs, maxSteps);
                obs = subEpochReturn.getLastObs();
                setStepCounter(getStepCounter() + subEpochReturn.getSteps());
                length += subEpochReturn.getSteps();
                rewards += subEpochReturn.getReward();
                double score = subEpochReturn.getScore();
                if (length >= this.getConf().getMaxEpochStep() || this.getMdp().isDone()) {
                    this.postEpoch();
                    DataManager.StatEntry statEntry = new AsyncThread.AsyncStatEntry(this.getStepCounter(), getEpochCounter(), rewards, length, score);
                    this.getDataManager().appendStat(statEntry);
                    System.out.println("ThreadNum-" + this.threadNumber + " Epoch: " + this.getEpochCounter() + ", reward: " + statEntry.getReward());
                    this.getCurrent().reset();
                    initMdp = Learning.initMdp(this.getMdp(), getHistoryProcessor());
                    obs = initMdp.getLastObs();
                    rewards = initMdp.getReward();
                    length = initMdp.getSteps();
                    setEpochCounter(getEpochCounter() + 1);
                    this.preEpoch();
                }
            }
        } catch (Exception var14) {
            this.getAsyncGlobal().setRunning(false);
            var14.printStackTrace();
        } finally {
            this.postEpoch();
        }

    }
}
