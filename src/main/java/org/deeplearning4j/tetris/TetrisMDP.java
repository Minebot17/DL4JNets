package org.deeplearning4j.tetris;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.mdp.ale.ALEMDP;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.DatagramPacket;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TetrisMDP implements MDP<TetrisMDP.InputData, Integer, DiscreteSpace> {

    //public static boolean sendPacket = false;
    protected static final Logger log = LoggerFactory.getLogger(ALEMDP.class);
    protected final ObservationSpace<InputData> observationSpace;
    protected final int[] balls = new int[]{ 0, 100, 300, 700, 1800 };
    protected boolean gameOver;
    protected boolean[] cells;
    protected Figure currentFigure;
    protected Figure nextFigure;
    protected int figureCount = 0;
    protected int leftCount = 0;
    protected int rightCount = 0;
    protected boolean isCopy = false;
    //protected float preReward = 0;
    //protected FileOutputStream file;
    //protected int rewardCounter = 0;

    public TetrisMDP(){
        observationSpace = new ArrayObservationSpace<>(new int[]{ 1, 200 });
        cells = new boolean[10*18];
        currentFigure = Figure.getRandomFigure();
        nextFigure = Figure.getRandomFigure();
    }

    // 0 - nothing, 1 - left, 2 - right, 3 - rotate | tick each 5 steps
    @Override
    public StepReply<InputData> step(Integer integer) {
        float preReward = getReward();
        Figure current = currentFigure;
        for (int i = 0; i < integer / 10; i++) {
            if (current != currentFigure)
                break;
            rotateAction();
            doTick();
        }
        if (integer % 10 < 5) {
            leftCount++;
            for (int i = 0; i < (integer % 10) + 1; i++) {
                if (current != currentFigure)
                    break;
                offsetAction(true);
                doTick();
            }
        }
        else {
            rightCount++;
            for (int i = 0; i < (integer % 10) - 4; i++) {
                if (current != currentFigure)
                    break;
                offsetAction(false);
                doTick();
            }
        }
        while (current == currentFigure && !gameOver)
            doTick();
        float currentReward = getReward();
        int lines = getLines(true);
        if (lines != 0)
            log.info(lines + " lines down");

        //rewardCounter += reward;
        if (gameOver)
            log.info("Left: " + (leftCount / (float)(leftCount + rightCount) * 100f) + "% Right: " + (rightCount / (float)(leftCount + rightCount) * 100f) + "%");
        return new StepReply(packInputData(), currentReward - preReward, gameOver, null);
    }

    protected void doTick(){
        boolean sendPacket = Tetris.unity;
        if (sendPacket && !isCopy) {
            try {
                byte[] toSend = new byte[180];
                for (int i = 0; i < 180; i++)
                    toSend[i] = cells[i] ? (byte) 1 : 0;
                for (int x = 0; x < 4; x++)
                    for (int y = 0; y < 4; y++)
                        if (currentFigure.getCell(x, y) && currentFigure.x + x >= 0 && currentFigure.y + y < 18 && currentFigure.y + y >= 0 && (currentFigure.x + x + (currentFigure.y + y) * 10) < 180)
                            toSend[currentFigure.x + x + (currentFigure.y + y) * 10] = 1;
                DatagramPacket packet = new DatagramPacket(toSend, toSend.length, Tetris.address, 5236);
                Tetris.socket.send(packet);
            } catch (Exception e) {
                e.printStackTrace();
            }

            try {
                Thread.sleep(15);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        if (isFigureOnGround(currentFigure)){
            if (currentFigure.y > 16 || currentFigure.y < -4){
                gameOver();
                return;
            }

            for(int x = 0; x < 4; x++)
                for(int y = 0; y < 4; y++) {
                    if (currentFigure.getCell(x, y) && currentFigure.y + y >= 0 && currentFigure.y + y < 18 && currentFigure.x + x >= 0 && currentFigure.x + x < 10)
                        setCell(currentFigure.x + x, currentFigure.y + y, currentFigure.getCell(x, y));
                    else if (currentFigure.getCell(x, y) && currentFigure.x + x >= 0 && currentFigure.x + x < 10) {
                        gameOver();
                        return;
                    }
                }
            /*if (figureCount >= 4){
                gameOver();
                return;
            }*/
            currentFigure = nextFigure;
            nextFigure = Figure.getRandomFigure();
            figureCount++;
            //log.info("Figure grounded");
        }
        currentFigure.y--;
        //return balls[lines] + (fg ? 1 : 0);
    }

    protected int getLines(boolean clear){
        int lines = 0;
        for(int y = 0; y < 18; y++) {
            boolean done = true;
            for (int x = 0; x < 10; x++)
                if (!getCell(x, y)){
                    done = false;
                    break;
                }

            if (done)
                lines++;

            if (clear && lines != 0)
                for(int x = 0; x < 10; x++)
                    setCell(x, y, y + lines < 18 && getCell(x, y + lines));
        }
        return lines;
    }

    protected float getReward(){
        float agrHeight = 0;
        float holes = 0;
        float bumpiness = 0;
        float[] heights = new float[10];
        float differ = 0;
        for(int x = 0; x < 10; x++) {
            for (int y = 17; y >= 0; y--)
                if (getCell(x, y)) {
                    agrHeight += y;
                    if (x != 0)
                        bumpiness += Math.abs(y - heights[x - 1]);
                    heights[x] = y;
                    break;
                }
            for (int y = 0; y < 17; y++)
                if (!getCell(x, y) && getCell(x, y))
                    holes++;
        }
        for(int x = 0; x < 10; x++)
            differ += Math.abs(heights[x] - agrHeight/10f);

        return -0.51f * agrHeight + 0.76f * getLines(false) + -0.8566f * holes + -0.1844f * bumpiness + -0.15f * differ;
    }

    protected void gameOver(){
        gameOver = true;
        //log.info("Game over");
        /*try {
            if (file == null) {
                if (!Files.exists(Paths.get("./rewardLog_" + Thread.currentThread().getName() + ".txt")))
                    Files.createFile(Paths.get("./rewardLog_" + Thread.currentThread().getName() + ".txt"));
                file = new FileOutputStream("./rewardLog_" + Thread.currentThread().getName() + ".txt");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            file.write((rewardCounter + "\n").getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }
        rewardCounter = 0;*/
    }

    protected void offsetAction(boolean left){
        if (!isSideNotEmpty(currentFigure, left))
            currentFigure.x += left ? -1 : 1;
    }

    protected void rotateAction(){
        Figure rotatedFigure = new Figure(currentFigure);
        rotatedFigure.rotate();

        for(int x = 0; x < 4; x++)
            for(int y = 0; y < 4; y++)
                if (rotatedFigure.getCell(x, y) && (rotatedFigure.y + y >= 18 || rotatedFigure.y + y < 0 || rotatedFigure.x + x < 0 || rotatedFigure.x + x >= 10 || getCell(rotatedFigure.x + x, rotatedFigure.y + y)))
                    return;

        currentFigure.rotate();
    }

    protected boolean isSideNotEmpty(Figure figure, boolean left){
        if (left) {
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    if (figure.getCell(x, y) && figure.y + y < 18 && figure.y + y >= 0) {
                        if (figure.x + x <= 0 || figure.x + x >= 10 || getCell(figure.x + x - 1, figure.y + y))
                            return true;
                        break;
                    }
        }
        else {
            for (int y = 0; y < 4; y++)
                for (int x = 3; x >= 0; x--)
                    if (figure.getCell(x, y) && figure.y + y < 18 && figure.y + y >= 0 && figure.x + x >= -1) {
                        if (figure.x + x >= 9 || getCell(figure.x + x + 1, figure.y + y))
                            return true;
                        break;
                    }
        }
                    
        return false;
    }

    protected boolean isFigureOnGround(Figure figure){
        for(int x = 0; x < 4; x++)
            for(int y = 0; y < 4; y++) {
                if (figure.y < -4)
                    return true;
                if (figure.getCell(x, y) && figure.y + y < 18 && figure.x + x >= 0 && figure.x + x < 10){
                    if (figure.y + y <= 0 || getCell(figure.x + x, figure.y + y - 1))
                        return true;
                    break;
                }
            }
        return false;
    }

    protected boolean getCell(int x, int y){
        return cells[x + y * 10];
    }

    protected void setCell(int x, int y, boolean value){
        cells[x + y * 10] = value;
    }

    @Override
    public ObservationSpace<InputData> getObservationSpace() {
        return observationSpace;
    }

    @Override
    public DiscreteSpace getActionSpace() {
        return new DiscreteSpace(40);
    }

    @Override
    public InputData reset() {
        cells = new boolean[10*18];
        currentFigure = Figure.getRandomFigure();
        nextFigure = Figure.getRandomFigure();
        gameOver = false;
        figureCount = 0;
        leftCount = 0;
        rightCount = 0;
        return packInputData();
    }

    @Override
    public void close() {
        /*try {
            file.close();
        } catch (IOException e) {
            e.printStackTrace();
        }*/
    }

    @Override
    public boolean isDone() {
        return gameOver;
    }

    @Override
    public MDP<InputData, Integer, DiscreteSpace> newInstance() {
        return new TetrisMDP();
    }

    public MDP<InputData, Integer, DiscreteSpace> copy(){
        TetrisMDP result = new TetrisMDP();
        result.gameOver = gameOver;
        result.cells = Arrays.copyOf(cells, 18*10);
        result.currentFigure = new Figure(currentFigure);
        result.nextFigure = new Figure(nextFigure);
        result.figureCount = figureCount;
        result.leftCount = leftCount;
        result.rightCount = rightCount;
        result.isCopy = true;
        return result;
    }

    // 180 - screen, 16 - current figure matrix, current index, next index, current x, current y | 200
    public InputData packInputData(){
        List<Double> data = new ArrayList<>();
        for(int x = 0; x < 10; x++)
            for(int y = 0; y < 18; y++)
                data.add(getCell(x, y) ? 1d : 0d);
        for(int x = 0; x < 4; x++)
            for(int y = 0; y < 4; y++)
                data.add(currentFigure.getCell(x, y) ? 1d : 0d);
        data.add((double) currentFigure.index);
        data.add((double) nextFigure.index);
        data.add((double) currentFigure.x);
        data.add((double) currentFigure.y);

        double[] result = new double[data.size()];
        for (int i = 0; i < data.size(); i++)
            result[i] = data.get(i);
        return new InputData(result);
    }

    public static class InputData implements Encodable {
        double[] data;

        public InputData(double[] data) {
            this.data = data;
        }

        public double[] toArray() {
            return this.data;
        }
    }
}
