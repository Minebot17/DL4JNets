package org.deeplearning4j;

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
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import sun.awt.image.ToolkitImage;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

public class Tracking {

    public static void main(String[] args) throws IOException {
        Scanner sc = new Scanner(System.in);
        System.out.println("Train new or load? [new/name]: ");
        String modelName = sc.nextLine();
        if (modelName.equals("new")) {
            Random rnd = new Random();
            int seed = rnd.nextInt();
            ImageIterator iterator = new ImageIterator(seed, 32, 32, 4, 24);
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new AdaDelta())
                    .weightInit(WeightInit.XAVIER)
                    .list(
                            new ConvolutionLayer.Builder(3, 3).padding(1, 1).nIn(3).nOut(16).hasBias(true).activation(Activation.RELU).build(),
                            new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build(), // 16
                            new ConvolutionLayer.Builder(3, 3).nIn(16).nOut(32).hasBias(true).activation(Activation.RELU).build(),
                            new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build(), // 7
                            new DenseLayer.Builder().nIn(1568).nOut(1000).hasBias(true).activation(Activation.SIGMOID).build(),
                            new DenseLayer.Builder().nIn(1000).nOut(500).hasBias(true).activation(Activation.SIGMOID).build(),
                            new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).nIn(500).nOut(20).activation(Activation.RELU).build()
                    ).setInputType(InputType.convolutionalFlat(32, 32, 3)).pretrain(false).backprop(true).build();
            MultiLayerNetwork network = new MultiLayerNetwork(conf);
            network.init();
            network.addListeners(new ScoreIterationListener(100));
            //network.fit(mnistDataSet);

            for (int e = 0; e < 10; e++) {
                for (int i = 0; i < 50000; i++) {
                    INDArray[] data = iterator.next();
                    float[] result = network.activate(data[0], Layer.TrainingMode.TEST).toFloatVector();
                    float[] notFlipped = Nd4j.concat(1, data[1], data[2]).toFloatVector();
                    float[] flipped = Nd4j.concat(1, data[2], data[1]).toFloatVector();

                    float notFlippedError = 0;
                    for (int j = 0; j < result.length; j++)
                        notFlippedError += Math.pow(notFlipped[j] - result[j], 2);

                    float flippedError = 0;
                    for (int j = 0; j < result.length; j++)
                        flippedError += Math.pow(flipped[j] - result[j], 2);

                    boolean flip = notFlippedError > flippedError;
                    network.fit(data[0], flip ? Nd4j.concat(1, data[2], data[1]) : Nd4j.concat(1, data[1], data[2]));
                }
                ModelSerializer.writeModel(network, "./last.nn", true);
                iterator.reset(seed);
                System.out.println("//////////////////");
                System.out.println("-----epoch " + e + "------");
                System.out.println("//////////////////");
            }

            //Evaluation eval = network.evaluate(lfwTestIterator);
            //System.out.println("Accuracy: " + eval.accuracy());

            ModelSerializer.writeModel(network, "./last.nn", true);
        }
        else {
            MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork("./" + modelName + ".nn");
            System.out.println("Continue? [y/n]");
            while (sc.nextLine().equals("y")) {
                ImageIterator iterator = new ImageIterator(3535, 32, 32, 4, 24);
                for (int i = 0; i < 10; i++) {
                    INDArray[] array = iterator.next();
                    BufferedImage image = ImageIO.read(new File("./imgs/last.png"));

                    INDArray result = network.activate(array[0], Layer.TrainingMode.TEST);
                    float[] out = result.toFloatVector();
                    for(int x = 0; x < image.getWidth(); x++)
                        for(int y = 0; y < image.getHeight(); y++)
                            if ((x == Math.round(out[0]) || x == Math.round(out[0] + out[2]) || y == Math.round(out[1]) || y == Math.round(out[1] + out[3])) && (x >= Math.round(out[0]) && x <= Math.round(out[0] + out[2]) && y >= Math.round(out[1]) && y <= Math.round(out[1] + out[3])))
                                image.setRGB(x, y, 0xff0000);

                    for(int x = 0; x < image.getWidth(); x++)
                        for(int y = 0; y < image.getHeight(); y++)
                            if ((x == Math.round(out[10]) || x == Math.round(out[10] + out[12]) || y == Math.round(out[11]) || y == Math.round(out[11] + out[13])) && (x >= Math.round(out[10]) && x <= Math.round(out[10] + out[12]) && y >= Math.round(out[11]) && y <= Math.round(out[11] + out[13])))
                                image.setRGB(x, y, 0x0000ff);

                    ImageIO.write(image, "png", new File("./imgs/out" + i + ".png"));
                /*float maxValue = 0;
                int maxIndex = -1;
                for (int i = 0; i < out.length; i++)
                    if (out[i] > maxValue){
                        maxValue = out[i];
                        maxIndex = i;
                    }

                System.out.println("Answer: " + maxIndex);*/
                }
                System.out.println("Continue? [y/n]");

            }
        }
    }

    public static class ImageIterator { // 0 - square, 1 - triangle, 2 - circle

        public Random rnd;
        public int width;
        public int height;
        public int minSize;
        public int maxSize;
        public int c;

        public ImageIterator(int seed, int width, int height, int minSize, int maxSize){
            reset(seed);
            this.width = width;
            this.height = height;
            this.maxSize = maxSize;
            this.minSize = minSize;
        }

        public void reset(int seed){
            rnd = new Random(seed);
            c = 0;
        }

        public INDArray[] next(){ // x, y, w, h, square, triangle, circle, r, g, b
            int firstW = rnd.nextInt(maxSize - minSize) + minSize;
            int firstH = rnd.nextInt(maxSize - minSize) + minSize;
            int firstShape = rnd.nextInt(3);
            if (firstShape == 2)
                firstH = firstW;
            float[] first = new float[]{
                    rnd.nextInt(width - firstW), rnd.nextInt(height - firstH),
                    firstW, firstH,
                    firstShape == 0 ? 1 : 0, firstShape == 1 ? 1 : 0, firstShape == 2 ? 1 : 0,
                    rnd.nextInt(256)/255f, rnd.nextInt(256)/255f, rnd.nextInt(256)/255f
            };

            int secondW = rnd.nextInt(maxSize - minSize) + minSize;
            int secondH = rnd.nextInt(maxSize - minSize) + minSize;
            int secondShape = rnd.nextInt(3);
            if (secondShape == 2)
                secondH = secondW;
            float[] second = new float[]{
                    rnd.nextInt(width - secondW), rnd.nextInt(height - secondH),
                    secondW, secondH,
                    secondShape == 0 ? 1 : 0, secondShape == 1 ? 1 : 0, secondShape == 2 ? 1 : 0,
                    rnd.nextInt(256)/255f, rnd.nextInt(256)/255f, rnd.nextInt(256)/255f
            };

            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            for(int x = 0; x < image.getWidth(); x++)
                for(int y = 0; y < image.getHeight(); y++)
                    image.setRGB(x, y, 0xffffff);
            addShape(image, first);
            addShape(image, second);
            smooth(image);
            try {
                ImageIO.write(image, "png", new File("./imgs/last.png"));
            } catch (IOException e) {
                e.printStackTrace();
            }

            float[][] imageData = new float[1][];
            imageData[0] = new float[image.getWidth() * image.getHeight() * 3];
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    imageData[0][y * image.getHeight() + x] = (255 - ((image.getRGB(x, y) >> 16) & 255)) / 255f;
                }
            }
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    imageData[0][y * image.getHeight() + x + image.getWidth()*image.getHeight()] = (255 - ((image.getRGB(x, y) >> 8) & 255)) / 255f;
                }
            }
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    imageData[0][y * image.getHeight() + x + image.getWidth()*image.getHeight()*2] = (255 - (image.getRGB(x, y) & 255)) / 255f;
                }
            }

            /*float[] union = new float[first.length + second.length];
            System.arraycopy(first, 0, union, 0, first.length);
            System.arraycopy(second, 0, union, first.length, second.length);
            c++;*/
            return new INDArray[]{ Nd4j.create(imageData), Nd4j.create(first), Nd4j.create(second) };
        }

        private void addShape(BufferedImage image, float[] data){
            if (data[4] == 1){
                for(int x = (int) data[0]; x < (int) (data[0] + data[2]); x++)
                    for(int y = (int) data[1]; y < (int) (data[1] + data[3]); y++)
                        image.setRGB(x, y, new Color((int) (data[7]*255f), (int) (data[8]*255f), (int) (data[9]*255f)).getRGB());
            }
            else if (data[5] == 1){
                for(int x = (int) data[0]; x < (int) (data[0] + data[2]); x++)
                    for(int y = (int) data[1]; y < (int) (data[1] + data[3]); y++)
                        if ((x - data[0])/data[2] + (y - data[1])/data[3] < 1)
                            image.setRGB(x, y, new Color((int) (data[7]*255f), (int) (data[8]*255f), (int) (data[9]*255f)).getRGB());
            }
            else if (data[6] == 1){
                for(int x = (int) data[0]; x < (int) (data[0] + data[2]); x++)
                    for(int y = (int) data[1]; y < (int) (data[1] + data[3]); y++)
                        if (Math.pow(x - (data[0] + data[2]/2f), 2) + Math.pow(y - (data[1] + data[3]/2f), 2) < data[2]/2f * data[3]/2f)
                            image.setRGB(x, y, new Color((int) (data[7]*255f), (int) (data[8]*255f), (int) (data[9]*255f)).getRGB());
            }
        }

        private void smooth(BufferedImage image){
            for(int x = 0; x < image.getWidth(); x++)
                for(int y = 0; y < image.getHeight(); y++){
                    int color = -1;
                    int error = 9999999;
                    int count = 0;
                    if (image.getRGB(x, y) == -1)
                        for(int x0 = -1; x0 <= 1; x0++)
                            for(int y0 = -1; y0 <= 1; y0++){
                                if (x + x0 < 0 || x + x0 >= image.getWidth() || y + y0 < 0 || y + y0 >= image.getHeight())
                                    continue;

                                int current = image.getRGB(x + x0, y + y0);
                                if (color != -1 && color == current)
                                    count++;
                                if (color == -1 && current != -1) {
                                    color = current;
                                    error = getColorError(current);
                                }
                                else if (color != -1 && color != current && getColorError(current) > error){
                                    color = current;
                                    error = getColorError(color);
                                }
                            }
                    if (color != -1)
                        image.setRGB(x, y, getGradientColor(0xffffff, color, 1/(1 + (float)Math.pow((float)Math.E, -count + 3))));
                }
        }

        private int getColorError(int color){
            int result = 0;
            result += Math.pow(255 - (color >> 16) & 255, 2);
            result += Math.pow(255 - (color >> 8) & 255, 2);
            result += Math.pow(255 - color & 255, 2);
            return result;
        }

        // Возвращает цвет между двумя цветами от 0 до 1. 0 - начальный цвет, 1 - конечный
        private int getGradientColor(int beginColor, int endColor, float percent){
            if (percent == 0)
                return beginColor;
            else if (percent == 1)
                return endColor;

            Vector3f beginVec = new Vector3f(beginColor);
            Vector3f endVec = new Vector3f(endColor);
            return beginVec.multiple(1 - percent).add(endVec.multiple(percent)).getColor();
        }
    }
}
