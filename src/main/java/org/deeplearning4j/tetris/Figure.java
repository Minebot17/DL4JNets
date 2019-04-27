package org.deeplearning4j.tetris;

import org.deeplearning4j.Vector3f;

public class Figure {
    protected static final Figure[] allFigures = new Figure[]{
        new Figure(new boolean[]{
                false,true,false,false,
                false,true,false,false,
                false,true,false,false,
                false,true,false,false },
                (byte) 1), // Line
        new Figure(new boolean[]{
                true,true,true,false,
                true,false,false,false,
                false,false,false,false,
                false,false,false,false },
                (byte) 0), // L left
        new Figure(new boolean[]{
                true,true,true,false,
                false,false,true,false,
                false,false,false,false,
                false,false,false,false },
                (byte) 0), // L right
        new Figure(new boolean[]{
                false,false,false,false,
                false,true,true,false,
                false,true,true,false,
                false,false,false,false },
                (byte) 2), // square
        new Figure(new boolean[]{
                false,false,false,false,
                true,true,false,false,
                false,true,true,false,
                false,false,false,false },
                (byte) 1), // S left
        new Figure(new boolean[]{
                false,false,false,false,
                false,true,true,false,
                true,true,false,false,
                false,false,false,false },
                (byte) 1), // S right
        new Figure(new boolean[]{
                false,true,false,false,
                true,true,true,false,
                false,false,false,false,
                false,false,false,false },
                (byte) 0), // pyramid
    };

    protected static int lastIndex;
    protected int index;
    protected boolean[] currentCells;
    protected byte rotateType; // 0 - full, 1 - half, 2 - not rotate
    protected byte currentRotate;

    public int x;
    public int y;

    protected Figure(boolean[] beginCells, byte rotateType){
        this.currentCells = beginCells;
        this.rotateType = rotateType;
        this.index = lastIndex;
        lastIndex++;
    }

    protected Figure(Figure figure){
        this.index = figure.index;
        this.currentCells = figure.currentCells;
        this.rotateType = figure.rotateType;
        this.currentRotate = figure.currentRotate;
        this.x = figure.x;
        this.y = figure.y;
    }

    public void rotate(){
        if (rotateType == 0){
            currentCells = rotateMatrix(true);
            currentRotate = currentRotate == 3 ? 0 : (byte) (currentRotate + 1);
        }
        else if (rotateType == 1){
            currentCells = rotateMatrix(currentRotate == 0);
            currentRotate = currentRotate != 0 ? 0 : (byte) 1;
        }
    }

    protected boolean[] rotateMatrix(boolean clockwise){
        boolean[] result = new boolean[16];
        if (clockwise){
            result[0] = currentCells[2];
            result[1] = currentCells[6];
            result[2] = currentCells[10];
            result[3] = currentCells[14];
            result[4] = currentCells[1];
            result[5] = currentCells[5];
            result[6] = currentCells[9];
            result[7] = currentCells[13];
            result[8] = currentCells[0];
            result[9] = currentCells[4];
            result[10] = currentCells[8];
            result[11] = currentCells[12];
        }
        else {
            result[0] = currentCells[8];
            result[1] = currentCells[4];
            result[2] = currentCells[0];
            result[4] = currentCells[9];
            result[5] = currentCells[5];
            result[6] = currentCells[1];
            result[8] = currentCells[10];
            result[9] = currentCells[6];
            result[10] = currentCells[2];
            result[12] = currentCells[11];
            result[13] = currentCells[7];
            result[14] = currentCells[3];
        }
        return result;
    }

    public boolean getCell(int x, int y){
        return currentCells[x + y * 4];
    }

    public void setCell(int x, int y, boolean value){
        currentCells[x + y * 4] = value;
    }

    public static Figure getRandomFigure(){
        Figure result = new Figure(allFigures[Tetris.rnd.nextInt(allFigures.length)]);
        int toRotate = Tetris.rnd.nextInt(4);
        for (int i = 0; i < toRotate; i++)
            result.rotate();
        result.x = Tetris.rnd.nextInt(6);
        result.y = 18;
        return result;
    }
}
