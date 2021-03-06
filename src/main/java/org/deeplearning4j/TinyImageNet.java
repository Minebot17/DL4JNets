package org.deeplearning4j;

import kotlin.text.Charsets;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

public class TinyImageNet {

    public static void main(String[] args) throws IOException {
        Scanner sc = new Scanner(System.in);
        System.out.println("Train new or load? [new/name]: ");
        String modelName = sc.nextLine();
        if (modelName.equals("new")) {
            TinyImageNetDataSetIterator lfwTrainIterator = new TinyImageNetDataSetIterator(10, DataSetType.TRAIN);
            TinyImageNetDataSetIterator lfwTestIterator = new TinyImageNetDataSetIterator(10, DataSetType.TEST);
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new AdaDelta())
                    .weightInit(WeightInit.XAVIER)
                    .list(
                            new ConvolutionLayer.Builder(3, 3).padding(1, 1).nIn(3).nOut(64).activation(Activation.RELU).build(),
                            new ConvolutionLayer.Builder(3, 3).nIn(64).nOut(128).activation(Activation.RELU).build(),
                            new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build(),
                            new ConvolutionLayer.Builder(3, 3).nIn(128).nOut(256).activation(Activation.RELU).build(),
                            new ConvolutionLayer.Builder(3, 3).nIn(256).nOut(256).activation(Activation.RELU).build(),
                            new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build(),
                            new ConvolutionLayer.Builder(3, 3).nIn(256).nOut(256).activation(Activation.RELU).build(),
                            new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build(),
                            new DenseLayer.Builder().nIn(6400).nOut(3200).hasBias(true).activation(Activation.SIGMOID).build(),
                            new DenseLayer.Builder().nIn(3200).nOut(200).hasBias(true).activation(Activation.SIGMOID).build(),
                            new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).nIn(200).nOut(200).activation(Activation.SIGMOID).build()
                    ).setInputType(InputType.convolutionalFlat(64, 64, 3)).pretrain(false).backprop(true).build();
            MultiLayerNetwork network = new MultiLayerNetwork(conf);
            network.init();
            network.addListeners(new ScoreIterationListener(10));
            //network.fit(mnistDataSet);

            EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                    .epochTerminationConditions(new MaxEpochsTerminationCondition(2))
                    .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(30, TimeUnit.MINUTES))
                    .scoreCalculator(new DataSetLossCalculator(lfwTestIterator, true))
                    .evaluateEveryNEpochs(1)
                    .modelSaver(new LocalFileModelSaver("./best", Charsets.UTF_8))
                    .saveLastModel(true)
                    .build();

            // training
            EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, network, lfwTrainIterator);
            EarlyStoppingResult result = trainer.fit();

            // print out early stopping results
            System.out.println("Termination reason: " + result.getTerminationReason());
            System.out.println("Termination details: " + result.getTerminationDetails());
            System.out.println("Total epochs: " + result.getTotalEpochs());
            System.out.println("Best epoch number: " + result.getBestModelEpoch());
            System.out.println("Score at best epoch: " + result.getBestModelScore());

            Evaluation eval = network.evaluate(lfwTestIterator);
            System.out.println("Accuracy: " + eval.accuracy());

            ModelSerializer.writeModel(network, "./last.nn", true);
        }
        else {
            MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork("./" + modelName + ".nn");
            System.out.println("Continue? [y/n]");
            while (sc.nextLine().equals("y")) {
                BufferedImage image = ImageIO.read(new File("./input.png"));
                float[][] result = new float[1][];
                result[0] = new float[64 * 64 * 3];
                for (int y = 0; y < image.getHeight(); y++) {
                    for (int x = 0; x < image.getWidth(); x++) {
                        result[0][y * 64 + x] = (255 - ((image.getRGB(x, y) >> 16) & 255)) / 255f;
                    }
                }
                for (int y = 0; y < image.getHeight(); y++) {
                    for (int x = 0; x < image.getWidth(); x++) {
                        result[0][y * 64 + x + 64*64] = (255 - ((image.getRGB(x, y) >> 8) & 255)) / 255f;
                    }
                }
                for (int y = 0; y < image.getHeight(); y++) {
                    for (int x = 0; x < image.getWidth(); x++) {
                        result[0][y * 64 + x + 64*64*2] = (255 - (image.getRGB(x, y) & 255)) / 255f;
                    }
                }

                INDArray array = network.activate(Nd4j.create(result), Layer.TrainingMode.TEST);
                float[] out = array.toFloatVector();

                float maxValue = 0;
                int maxIndex = -1;
                for (int i = 0; i < out.length; i++)
                    if (out[i] > maxValue){
                        maxValue = out[i];
                        maxIndex = i;
                    }

                System.out.println("Answer: " + maxIndex);
                System.out.println("Continue? [y/n]");
            }
        }
    }
}
