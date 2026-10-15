package com.aufy.jnet.core.backend.interfaces.composite;

/** 
 * Applies a full array mapping into an array expanded with the oroginal double value.
 * This interface is a more convenient wrapper for Mapping.apply from which this interface is derived.
 * 
 * A java equivalent of the method is provided below:
 * 
 * {@snippet :
 * double[] result = new double[newSize];
 * Arrays.fill(result, data);
 * return operation(result);
 * }
 * 
 * @param data an array of double values
 * @param newSize the new size of the array
 * @param operation a unary operation (accepts one double and returns a double) to apply to each element, can either be a lambda expression or a reference to a method
 * @return a new array containing the results of applying the operation to each element of the input array
 */
@FunctionalInterface
public interface Expansion {
  double[] _apply(double inputData);

}
