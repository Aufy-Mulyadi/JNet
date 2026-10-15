package com.aufy.jnet.core.backend.interfaces.composite;

/**
 * Applies a binary operation to all elements of an array, but updates an accumulator based on a starting value.
 * This interface is a more convenient wrapper for AccumilatorOp.apply from which this interface is derived.
 * 
 * {@snippet :
 * double result = initial;
 * for (int i = 0; i < data.length; i++) {
 *   result = operation(result, data[i]);
 * }
 * return result;
 * }
 * 
 * @param data an array of double values.
 * @param operation an accumulator operation to apply to parse all elements, can either be a lambda expression or a reference to a method.
 * @return a double of parsing all the elements and applying the operation to an output accumulator.
 */
@FunctionalInterface
public interface Reduction {
  double _apply(double[] inputData);

  /**
   * Applies a full reduction to all elements of an array.
   * 
   * @param data an array of double values.
   * @param operation a Reduction operation to apply to all elements, can either be a lambda expression or a reference to a method.
   * @return a double to the output of the accumulator.
   */
  public default double apply(double[] data) {
    return _apply(data);
  }

}
