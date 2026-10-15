package com.aufy.jnet.core.backend.exceptions.statistics;

import com.aufy.jnet.core.backend.exceptions.Message;

/**
 * Checks if a value is within bounds.
 */
public class General {
  /**
   * Checks if a double is a proper probability. Must be between 0 and 1 inclusive.
   * 
   * @param data the data to check
   */
  public static void isProbability(double data) {
    if (data < 0 || data > 1) Message.crash("statistical", "Illegal probability", null, "a proper probability must be between 0 and 1 inclusive");
  }

}
