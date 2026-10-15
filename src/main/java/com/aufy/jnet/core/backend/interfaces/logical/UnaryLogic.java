package com.aufy.jnet.core.backend.interfaces.logical;

/** 
 * Applies a boolean unary operation to all elements of an array.
 * 
 * {@snippet :
 * double[] result = new double[data.length];
 * for (int i = 0; i < data.length; i++) {
 *   result[i] = operation(data[i]) ? 1 : 0;
 * }
 * return result;
 * }
 * 
 * @param data an array of double values
 * @param operation a unary operation (accepts one double and returns a boolean) to apply to each element, can either be a lambda expression or a reference to a method
 * @return a new array containing the results of applying the boolean operation to each element of the input array
 */
@FunctionalInterface
public interface UnaryLogic {
  boolean _apply(double data);

  public static double[] apply(double[] data, UnaryLogic operation) {
    double[] result = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      result[i] = operation._apply(data[i]) ? 1 : 0;
    }
    return result;
  }
}
