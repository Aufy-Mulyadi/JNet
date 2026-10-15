package com.aufy.jnet.core.backend.interfaces.composite;

/** 
 * Applies a unary operation to all elements of an array.
 * This interface is a more convenient wrapper for UnaryOp.apply from which this interface is derived.
 * 
 * A java equivalent of the method is provided below:
 * 
 * {@snippet :
 * double[] result = new double[data.length];
 * for (int i = 0; i < data.length; i++) {
 *   result[i] = operation(data[i]);
 * }
 * return result;
 * }
 * 
 * @param data an array of double values
 * @param operation a unary operation (accepts one double and returns a double) to apply to each element, can either be a lambda expression or a reference to a method
 * @return a new array containing the results of applying the operation to each element of the input array
 */
@FunctionalInterface
public interface Mapping {
  double[] _apply(double[] inputData);

  /**
   * Applies a full array mapping to all elements of an array.
   * 
   * @param data an array of double values.
   * @param operation a Mapping operation to apply to all elements, can either be a lambda expression or a reference to a method.
   * @return a double to the output of the mapping.
   */
  public default double[] apply(double[] data) {
    return _apply(data);
  }

  public static double[] one(double[] value) {
    double[] result = new double[value.length];
    for (int i = 0; i < value.length; i++) result[i] = 1;
    return result;
  }

  public static double[] zero(double[] value) {
    double[] result = new double[value.length];
    return result;
  }
}
