package com.aufy.jnet.core.backend.exceptions.statistics;

import com.aufy.jnet.core.backend.exceptions.Message;

/**
 * Checks if a value is within bounds.
 */
public class Parameter {
  /**
   * Checks if a double is positive non-zero or not.
   * 
   * @param data the data to check
   */
  public static void isPositive(double data) {
    if (data <= 0) Message.crash("statistical initialization", "Illegal parameter", null, "Parameter cannot be negative or zero");
  }

  /**
   * Checks if a double is above a cutoff.
   * 
   * @param cutoff the cutoff to compare if data is above or not
   * @param data the data to check
   */
  public static void isAboveValue(double cutoff, double data, String distribution) {
    if (data < cutoff) Message.crash("statistical initialization", "Illegal parameter", distribution , "parameter must be above " + cutoff);
  }

}
