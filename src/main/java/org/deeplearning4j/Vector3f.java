package org.deeplearning4j;

public class Vector3f {
    public float x;
    public float y;
    public float z;

    public Vector3f(float x, float y, float z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public Vector3f(int color){
        x = (color >> 16) & 255;
        y = (color >> 8) & 255;
        z = color & 255;
    }

    public float getLength(){
        return (float) Math.sqrt(x*x + y*y + z*z);
    }

    public Vector3f add(float x, float y, float z){
        return add(new Vector3f(x, y, z));
    }

    public Vector3f add(float value){
        return add(new Vector3f(x + value, y + value, z + value));
    }

    public Vector3f add(Vector3f vector) {
        return new Vector3f(x + vector.x, y + vector.y, z + vector.z);
    }

    public Vector3f subtract(Vector3f vector){
        return add(vector.invert());
    }

    public Vector3f invert(){
        return new Vector3f(-x, -y, -z);
    }

    public Vector3f multiple(float value){
        return new Vector3f(x * value, y * value, z * value);
    }

    public Vector3f multiple(Vector3f vector){
        return new Vector3f(x * vector.x, y * vector.y, z * vector.z);
    }

    public Vector3f divide(float value){
        return new Vector3f(x / value, y / value, z / value);
    }

    public Vector3f divide(Vector3f vector){
        return new Vector3f(x / vector.x, y / vector.y, z / vector.z);
    }

    public float getDistance(Vector3f vector){
        return vector.subtract(this).getLength();
    }

    public Vector3f round(){
        return new Vector3f(Math.round(x), Math.round(y), Math.round(z));
    }

    public Vector3f normalize(){
        return divide(getLength());
    }

    public int getColor(){
        return ((int)x << 16) | ((int)y << 8) | (int)z;
    }

    @Override
    public String toString(){
        return "x: " + x + " y: " + y + " z: " + z;
    }

    public Vector3f copy(){
        return new Vector3f(x, y, z);
    }
}
