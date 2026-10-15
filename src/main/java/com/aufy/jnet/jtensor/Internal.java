package com.aufy.jnet.jtensor;

public class Internal {
  public Tensor tensor;

  public double[] data;
  public int[] strides;
  public int[] shape;
  public int rank;
  public int size; 

  public Internal(Tensor tensor) {
    this.tensor = tensor;

    this.data = tensor.core.core.data;
    this.strides = tensor.core.core.strides;
    this.shape = tensor.core.core.shape;
    this.rank = tensor.core.core.rank;
    this.size = tensor.core.core.size;
  }

  @Override
  public String toString() {
    if (tensor == null) {
      return "Null";
    } else {
      return "Internal(" + tensor + ")";
    }
  }

  public void setStrides(int[] strides) {
    System.arraycopy(strides, 0, tensor.core.core.strides, 0, strides.length);
    this.strides = tensor.core.core.strides;
  }

  public static void setStrides(Tensor tensor, int[] strides) {
    System.arraycopy(strides, 0, tensor.core.core.strides, 0, strides.length);
  }

  public static int[] getStrides(Tensor tensor) {
    return tensor.core.core.strides;
  }

  public void setShape(int[] shape) {
    System.arraycopy(shape, 0, tensor.core.core.shape, 0, shape.length);
  }

  public static void setShape(Tensor tensor, int[] shape) {
    System.arraycopy(shape, 0, tensor.core.core.shape, 0, shape.length);
  }

  public static int[] getShape(Tensor tensor) {
    return tensor.core.core.shape;
  }

  public void setData(double[] data) {
    System.arraycopy(data, 0, tensor.core.core.data, 0, data.length);
    this.data = tensor.core.core.data;
  }

  public static void setData(Tensor tensor, double[] data) {
    System.arraycopy(data, 0, tensor.core.core.data, 0, data.length);
  }

  public static double[] getData(Tensor tensor) {
    return tensor.core.core.data;
  }
}
