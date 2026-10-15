package com.aufy.jnet.jtensor;

import java.util.Arrays;
import java.util.Random;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.arrayops.Tools;
import com.aufy.jnet.core.backend.exceptions.statistics.Parameter;

public class Init {
  private static final Random RNG = new Random();
  private final int[] shape;
  private final int size;
  private double[] data;
  
  public Init(int... shape) {
    this.data = new double[Reductions.prod(shape)];
    this.shape = shape;
    this.size = Reductions.prod(shape);
  }
  
  public Init(Tensor tensor) {
    this(tensor.shape);
  }

  @Override
  public String toString() {
    if (shape.length == 0) {
      return "UninitializedTensor[]()";
    } else {
      return "UninitializedTensor" + Arrays.toString(shape) + "()";
    }
  }

  // ###########################################################################################################
  // IMPLEMENTATIONS
  // ###########################################################################################################

  /**
   * Arranges the elements of the tensor in values from 0 to (size - 1).
   * 
   * @return a tensor with values from 0 to (size - 1).
   */
  public Tensor arrange() {
    return new Tensor(Tools.arrangeDouble(Reductions.prod(shape)), shape);
  }

  /**
   * Fills the tensor with a constant value.
   * 
   * @param value the value to fill the tensor with.
   * @return a tensor filled with the given value.
   */
  public Tensor fill(double value) {
    for (int i = 0; i < size; i++) data[i] = value;

    return new Tensor(data, shape);
  }

  /**
   * Fills the tensor with zeros.
   * 
   * @return a tensor filled with zeros.
   */
  public Tensor zeros() {
    return new Tensor(data, shape);
  }

  /**
   * Fills the tensor with zeros.
   * 
   * @return a tensor filled with ones.
   */
  public Tensor ones() {
    for (int i = 0; i < size; i++) data[i] = 1;

    return new Tensor(data, shape);
  }

  /**
   * Fills the tensor with random values from an exponential distribution.
   * 
   * @param lambda the rate parameter.
   * @return a tensor filled with random values from an exponential distribution.
   */
  public Tensor exponential(double lambda) {
    Parameter.isPositive(lambda);

    for (int i = 0; i < size; i++) data[i] = RNG.nextExponential() / lambda;

    return new Tensor(data, shape);
  }

  /**
   * Fills the tensor with random values from a normal (Gaussian) distribution.
   * 
   * @param mean the mean.
   * @param std the standard deviation.
   * @return a tensor filled with random values from a normal (Gaussian) distribution.
   */
  public Tensor normal(double mean, double std) {
    Parameter.isPositive(std);

    for (int i = 0; i < size; i++) data[i] = mean + std * RNG.nextGaussian();

    return new Tensor(data, shape);
  }

  /**
   * Fills the tensor with random values from a uniform distribution.
   * 
   * @param lower the lower bound.
   * @param upper the upper bound.
   * @return a tensor filled with random values from a uniform distribution.
   */
  public Tensor uniform(double lower, double upper) {
    Parameter.isAboveValue(upper, lower, "Uniform");

    for (int i = 0; i < size; i++) data[i] = lower + (upper - lower) * RNG.nextDouble();

    return new Tensor(data, shape);
  }
}
