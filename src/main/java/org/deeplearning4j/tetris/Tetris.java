package org.deeplearning4j.tetris;

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteConv;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteDense;
import org.deeplearning4j.rl4j.mdp.ale.ALEMDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdConv;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdDense;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.NoOp;

import java.io.IOException;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class Tetris {
    public static DatagramSocket socket;
    public static InetAddress address;
    public static Random rnd = new Random();
    public static final boolean unity = true;

    public static A3CDiscrete.A3CConfiguration ALE_A3C =
            new A3CDiscrete.A3CConfiguration(
                    123,            //Random seed
                    800000,          //Max step By epoch
                    10000,        //Max step
                    4,              //Number of threads
                    5000000,             //t_max
                    0,            //num step noop warmup
                    1,            //reward scaling
                    0.99,           //gamma
                    10.0            //td-error clipping
            );

    public static final ActorCriticFactoryCompGraphStdDense.Configuration ALE_NET_A3C =
            new ActorCriticFactoryCompGraphStdDense.Configuration(
                    4, 256,
                    0.001,   //l2 regularization
                    new Nesterovs(0.001f, 0.1f), //learning rate
                    null, true
            );

    public static void main(String[] args) throws IOException {
        ALE_A3C.setNumThread(unity ? 0 : 4);

        socket = new DatagramSocket();
        address = InetAddress.getByName("127.0.0.1");

        DataManager manager = new DataManager(true);
        TetrisMDP mdp = new TetrisMDP();

        /*A3CDiscreteDense<TetrisMDP.InputData> a3c = null;
        a3c = new A3CDiscreteDense<>(mdp,
                ACPolicy.load("./testModel2").getNeuralNet()
                //ALE_NET_A3C
                , ALE_A3C, manager);
        //a3c.setHistoryProcessor((IHistoryProcessor) null);
        if (unity)
            for (int i = 0; i < 10; i++)
                a3c.train();
        else
            a3c.train();
        a3c.getPolicy().save("./testModel2");*/

        while (true) {
            List<Double> results = new ArrayList<>();
            for (int i = 0; i < 40; i++)
                results.add(mdp.copy().step(i).getReward());
            int max = -1;
            double maxValue = -9999;
            for (int i = 0; i < 40; i++)
                if (results.get(i) > maxValue) {
                    maxValue = results.get(i);
                    max = i;
                }
            mdp.step(max);
            if (mdp.gameOver)
                mdp.reset();
        }

        //mdp.close();
        //socket.close();
    }
}
