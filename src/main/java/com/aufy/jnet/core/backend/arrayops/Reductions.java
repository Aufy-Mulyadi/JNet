package com.aufy.jnet.core.backend.arrayops;

import com.aufy.jnet.core.backend.interfaces.atomic.AccumulatorOp;
import com.aufy.jnet.core.backend.interfaces.atomic.UnaryOp;

/**
 * Manages raw mathematical operations for both integer and double arrays.
 * 
 * <p>
 * This class provide simple bare-bones mathematical operations for integer and double arrays. All double implementations rely on
 * {@link com.aufy.jnet.core.backend.interfaces.atomic.AccumulatorOp} as a baseline tool.
 * <p>
 * 
 */
public class Reductions {
  
  /**
   * Sums the entire array.
   * 
   * @param array array.
   * @return sum.
   */
  public static double sum(double[] array) {
    return AccumulatorOp.apply(array, 0, (a, b) -> a + b);
  }

  /**
   * Sums the entire array.
   * 
   * @param array array.
   * @return sum.
   */
  public static int sum(int[] array) {
    int output = 0;
    for (int n : array) output += n;
    return output;
  }

  /**
   * multiplies the entire array.
   * 
   * @param array array.
   * @return product.
   */
  public static double prod(double[] array) {
    return AccumulatorOp.apply(array, 1, (a, b) -> a * b);
  }

  /**
   * multiplies the entire array.
   * 
   * @param array array.
   * @return product.
   */
  public static int prod(int[] array) {
    int output = 1;
    for (int n : array) output *= n;
    return output;
  }

  /**
   * Find the minimum value in an array.
   * 
   * @param array array.
   * @return minimum value.
   */
  public static double min(double[] array) {
    return AccumulatorOp.apply(array, Double.MAX_VALUE, (a, b) -> (b < a) ? b : a);
  }

  /**
   * Find the maximum value in an array.
   * 
   * @param array array.
   * @return maximum value.
   */
  public static double max(double[] array) {
    return AccumulatorOp.apply(array, -Double.MAX_VALUE, (a, b) -> Math.max(a, b));
  }

  /**
   * Find the maximum value in an array.
   * 
   * @param array array.
   * @return maximum value.
   */
  public static int max(int[] array) {
    int output = Integer.MIN_VALUE;
    for (int n : array) output = (n > output) ? n : output;
    return output;
  }

  /**
   * Find the dot product of two arrays.
   * 
   * @param array1 the first array.
   * @param array2 the second array.
   * @return dot product.
   */
  public static int dot(int[] array1, int[] array2) {
    return (int) sum(Elementwise.multiply(array1, array2));
  }

  /**
   * Computes the log sum of an array of exponentials.
   * 
   * @param array input array.
   * @return log sum of exponentials.
   */
  public static double logSumExp(double[] array) {
    double max = max(array);
    double[] shifted = Uniform.subtract(array, max);

    double sum = sum(UnaryOp.apply(shifted, a -> Math.exp(a)));

    return Math.log(sum) + max;
  }

  /**
   * Computes the index of the maximum value.
   * 
   * @param array input array.
   * @return index of the maximum value.
   */
  public static double argmax(double[] array) {
    return Tools.indexOf(array, max(array));
  }

  /**
   * Computes the index of the minimum value.
   * 
   * @param array input array.
   * @return index of the minimum value.
   */
  public static double argmin(double[] array) {
    return Tools.indexOf(array, min(array));
  }

}
