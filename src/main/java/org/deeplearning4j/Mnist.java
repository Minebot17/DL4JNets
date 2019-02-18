package org.deeplearning4j;

import com.google.flatbuffers.FlatBufferBuilder;
import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.file.FileDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.SingletonDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.list.IntNDArrayList;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

public class Mnist {

    public static void main(String[] args) throws IOException {
        Scanner sc = new Scanner(System.in);
        System.out.println("Train new or load? [new/name]: ");
        String modelName = sc.nextLine();
        if (modelName.equals("new")) {
            MnistDataSetIterator mnistDataSet = new MnistDataSetIterator(16, 16000);
            MnistDataSetIterator mnistTest = new MnistDataSetIterator(16, 1000);
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Nesterovs(0.1d, 0.5d))
                    .weightInit(WeightInit.XAVIER)
                    .list(
                            new ConvolutionLayer.Builder(3, 3).padding(1, 1).nIn(1).nOut(10).hasBias(true).activation(Activation.RELU).build(),
                            new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build(),
                            new ConvolutionLayer.Builder(3, 3).nIn(10).nOut(20).hasBias(true).activation(Activation.RELU).build(),
                            new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build(),
                            new DenseLayer.Builder().nIn(720).nOut(480).hasBias(true).activation(Activation.RELU).build(),
                            new DenseLayer.Builder().nIn(480).nOut(320).hasBias(true).activation(Activation.RELU).build(),
                            new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).nIn(320).nOut(10).activation(Activation.SOFTMAX).build()
                    ).setInputType(InputType.convolutionalFlat(28, 28, 1)).pretrain(false).backprop(true).build();
            MultiLayerNetwork network = new MultiLayerNetwork(conf);
            network.init();
            network.addListeners(new ScoreIterationListener(16));
            //network.fit(mnistDataSet);

            EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                    .epochTerminationConditions(new MaxEpochsTerminationCondition(15))
                    .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
                    .scoreCalculator(new DataSetLossCalculator(mnistTest, true))
                    .evaluateEveryNEpochs(1)
                    .build();

            // training
            EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, network, mnistDataSet);
            EarlyStoppingResult result = trainer.fit();

            // print out early stopping results
            System.out.println("Termination reason: " + result.getTerminationReason());
            System.out.println("Termination details: " + result.getTerminationDetails());
            System.out.println("Total epochs: " + result.getTotalEpochs());
            System.out.println("Best epoch number: " + result.getBestModelEpoch());
            System.out.println("Score at best epoch: " + result.getBestModelScore());

            Evaluation eval = network.evaluate(mnistTest);
            System.out.println("Accuracy: " + eval.accuracy());

            ModelSerializer.writeModel(network, "./last.nn", true);
        }
        else {
            MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork("./" + modelName + ".nn");
            System.out.println("Continue? [y/n]");
            while (sc.nextLine().equals("y")) {
                BufferedImage image = ImageIO.read(new File("./input.png"));
                float[][] result = new float[1][];
                result[0] = new float[28 * 28];
                for (int y = 0; y < image.getHeight(); y++) {
                    for (int x = 0; x < image.getWidth(); x++) {
                        result[0][y * 28 + x] = (255 - (image.getRGB(x, y) & 255)) / 255f;
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
