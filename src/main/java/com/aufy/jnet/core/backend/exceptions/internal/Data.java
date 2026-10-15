package com.aufy.jnet.core.backend.exceptions.internal;

import com.aufy.jnet.core.backend.exceptions.Message;

/**
 * Checks for invalid data in data (array or scalar).
 * 
 */
public class Data {
  /*
  all errors here should deal with the double[] data directly. the user would most likely clearer and friendlier errors set in graph
  up the chain anyways, so errors here should be ominous and look like they are coming from the depths of hell
  */

  /**
   * Check if a double[] data contains any invalid values (NaN or Infinity).
   * 
   * @param data the double[] data to check
   */
  public static void verifyData(double[] data) {
    if (data == null) Message.crash("internal", "Compromised data", null, "data is null");
    if (data.length == 0) Message.crash("internal", "Compromised data", null, "data is empty");
    for (double n : data) {
      if (Double.isNaN(n)) Message.crash("internal", "Bad data", null, "data contains NaN");
      if (Double.isInfinite(n)) Message.crash("internal", "Bad data", null, "data contains Infinity");
    }
  }

  /**
   * Check if a Object[] data contains any null values, is null itself, or is empty.
   * 
   * @param data the Object[] data to check
   */
  public static void verifyData(Object[] data) {
    if (data == null) Message.crash("internal", "Compromised data", null, "data is null");
    if (data.length == 0) Message.crash("internal", "Compromised data", null, "data is empty");
    for (Object obj : data) {
      if (obj == null) Message.crash("internal", "Bad data", null, "data contains NaN");
    }
  }

  /**
   * Check if an object is null.
   * 
   * @param data the object to check
   */
  public static void verifyData(Object data) {
    if (data == null) Message.crash("internal", "Compromised data", null, "data is null");
  }

  /**
   * Ensures if two heaps have the same memory size.
   * 
   * @param dataA first heap to check
   * @param dataB second heap to check
   */
  public static void verifyEqualSizes(String operationName, double[] dataA, double[] dataB) {
    if (dataA.length != dataB.length) Message.crash("internal", "Incompatible data", operationName, "binary operation halted because of differing memory buffer sizes (" + dataA.length + " and " + dataB.length + ")");
  }

  /**
   * Ensures if a heap is above a limit. If its not, then it will throw an exception.
   * 
   * @param cutoff the cutoff to compare if data is above or not.
   * @param data the data to check.
   * @param operation the name of the calling operation.
   */
  public static void aboveValue(double cutoff, double[] data, String operation) {
    for (double n: data) if (n < cutoff) Message.crash("tensor data", "Illegal data values", operation , "tensor data must be at least " + cutoff + " but one of the values is " + n);
  }

  /**
   * Ensures if a heap is below a limit. If its not, then it  will throw an exception.
   * 
   * @param cutoff the cutoff to compare if data is below or not.
   * @param data the data to check.
   * @param operation the name of the calling operation.
   */
  public static void belowValue(double cutoff, double[] data, String operation) {
    for (double n: data) if (n > cutoff) Message.crash("tensor data", "Illegal data values", operation , "tensor data cannot exceed " + cutoff + " but one of the values is " + n);
  }

  /**
   * Ensures if a heap is between two values. If not, then it will throw an exception.
   * 
   * @param lower the lower inclusive bound.
   * @param upper the upper inclusive bound.
   * @param data the data to check.
   * @param operation the name of the calling operation.
   */
  public static void betweenValue(double lower, double upper, double[] data, String operation) {
    for (double n: data) if (n > upper || n < lower) Message.crash("tensor data", "Illegal data values", operation , "tensor data must be between " + lower + " and " + upper + " but one of the values is " + n);
  }
}
