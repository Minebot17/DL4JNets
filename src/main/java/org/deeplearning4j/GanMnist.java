package org.deeplearning4j;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

public class GanMnist {

    public static void main(String[] args) throws IOException {
        MnistDataSetIterator mnistDataSet = new MnistDataSetIterator(10, 16000);

        MultiLayerConfiguration discriminatorConf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.01f, 0.5f))
                .weightInit(WeightInit.XAVIER)
                .list(
                        new ConvolutionLayer.Builder(3, 3).nIn(1).nOut(32).stride(2, 2).padding(1, 1).hasBias(true).activation(Activation.LEAKYRELU).build(), // to 14
                        new BatchNormalization.Builder().build(),
                        new ConvolutionLayer.Builder(3, 3).nIn(32).nOut(64).stride(2, 2).padding(1, 1).hasBias(true).activation(Activation.LEAKYRELU).build(), // to 7
                        new BatchNormalization.Builder().build(),
                        new DenseLayer.Builder().nIn(3136).nOut(3136).hasBias(true).activation(Activation.LEAKYRELU).build(),
                        new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).nIn(3136).nOut(1).activation(Activation.SIGMOID).build()
                ).setInputType(InputType.convolutionalFlat(28, 28, 1)).pretrain(false).backprop(true).build();

        MultiLayerConfiguration generatorConf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.01f, 0.5f))
                .weightInit(WeightInit.XAVIER)
                .list(
                        new DenseLayer.Builder().nIn(3136).nOut(0).hasBias(true).activation(Activation.LEAKYRELU).build(),
                        new Deconvolution2D.Builder(3, 3).nIn(64).nOut(32).stride(2, 2).padding(1, 1).hasBias(true).activation(Activation.LEAKYRELU).build(), // to 14
                        new BatchNormalization.Builder().build(),
                        new Deconvolution2D.Builder(3, 3).nIn(32).nOut(1).stride(2, 2).padding(1, 1).hasBias(true).activation(Activation.LEAKYRELU).build(), // to 28
                        new BatchNormalization.Builder().build(),
                        new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).activation(Activation.TANH).build()
                ).setInputType(InputType.feedForward(3136)).pretrain(false).backprop(true).build();

        MultiLayerNetwork discriminator = new MultiLayerNetwork(discriminatorConf);
        discriminator.init();
        discriminator.addListeners(new ScoreIterationListener(10));

        MultiLayerNetwork generator = new MultiLayerNetwork(generatorConf);
        generator.init();
        generator.addListeners(new ScoreIterationListener(10));


    }
}
