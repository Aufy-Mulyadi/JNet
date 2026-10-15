package com.aufy.jnet.jtensor.inplace;

import java.util.Arrays;
import java.util.Random;

import com.aufy.jnet.core.backend.exceptions.statistics.Parameter;
import com.aufy.jnet.jtensor.Internal;
import com.aufy.jnet.jtensor.Tensor;

public class Fill {
  private static final Random RNG = new Random();
  private final Internal internal;
  private final int[] shape;
  private final double[] data;
  private final int size;
  
  public Fill(Tensor tensor) {
    this.internal = new Internal(tensor);
    this.data = internal.data;
    this.shape = internal.shape;
    this.size = internal.size;
  }

  @Override
  public String toString() {
    if (data.length == 0) {
      return "Null";
    } else {
      return "ReferenceTensor" + Arrays.toString(shape) + "()";
    }
  }

  // ###########################################################################################################
  // IMPLEMENTATIONS
  // ###########################################################################################################

  /**
   * Overides the data of this tensor.
   * 
   * @return a reference to this tensor.
   */
  public Tensor set(double[] data) {
    internal.setData(data);
    return internal.tensor;
  }

  /**
   * Randomly arranges the data from 0 to (size - 1).
   * 
   * @return a reference to this tensor.
   */
  public Tensor arrange() {
    for (int i = 0; i < size; i++) data[i] = i;
    internal.setData(data);
    return internal.tensor;
  }

  /**
   * Fills this tensor with a constant value.
   * 
   * @param value the value to fill the tensor with.
   * @return a reference to this tensor.
   */
  public Tensor fill(double value) {
    for (int i = 0; i < size; i++) data[i] = value;
    internal.setData(data);
    return internal.tensor;
  }
  
  /**
   * Fills this tensor with zeros.
   * 
   * @return a reference to this tensor.
   */
  public Tensor zeros() {
    for (int i = 0; i < size; i++) data[i] = 0.0;
    internal.setData(data);
    return internal.tensor;
  }
  
  /**
   * Fills this tensor with ones.
   * 
   * @return a reference to this tensor.
   */
  public Tensor ones() {
    for (int i = 0; i < size; i++) data[i] = 1;
    internal.setData(data);
    return internal.tensor;
  }
  
  /**
   * Fills this tensor with random values from an exponential distribution.
   * 
   * @param lambda the rate parameter.
   * @return a reference to this tensor.
   */
  public Tensor exponential(double lambda) {
    Parameter.isPositive(lambda);
    
    for (int i = 0; i < size; i++) data[i] = RNG.nextExponential() / lambda;
    internal.setData(data);
    return internal.tensor;
  }
  
  /**
   * Fills this tensor with random values from a normal distribution.
   * 
   * @param mean the mean.
   * @param std the standard deviation.
   * @return a reference to this tensor.
   */
  public Tensor normal(double mean, double std) {
    Parameter.isPositive(std);
    
    for (int i = 0; i < size; i++) data[i] = mean + std * RNG.nextGaussian();
    internal.setData(data);
    return internal.tensor;
  }
  
  /**
   * Fills this tensor with random values from a uniform distribution.
   * 
   * @param lower the lower bound.
   * @param upper the upper bound.
   * @return a reference to this tensor.
   */
  public Tensor uniform(double lower, double upper) {
    Parameter.isAboveValue(upper, lower, "Uniform");
    
    for (int i = 0; i < size; i++) data[i] = lower + (upper - lower) * RNG.nextDouble();
    internal.setData(data);
    return internal.tensor;
  }
}
