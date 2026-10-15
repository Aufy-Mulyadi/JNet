package com.aufy.jnet.core.backend.arrayops;

import java.util.Arrays;
import java.util.HashMap;

import com.aufy.jnet.core.backend.exceptions.statistics.General;

/**
 * Utility class for calculating order statistics, central tendency, and vector norms.
 */
public class Statistics {

  /**
   * Finds the minimum value in the array.
   *
   * @param array the input data
   * @return the smallest value
   */
  public static double min(double[] array) {
    double ans = Double.MAX_VALUE;
    for (double n : array) ans = (n < ans) ? n : ans;
    return ans;
  }

  /**
   * Finds the maximum value in the array.
   *
   * @param array the input data
   * @return the largest value
   */
  public static double max(double[] array) {
    double ans = -Double.MAX_VALUE;
    for (double n : array) ans = (n > ans) ? n : ans;
    return ans;
  }

  /**
   * Identifies the most frequently occurring value in the array.
   *
   * @param array the input data
   * @return the mode value
   */
  public static double mode(double[] array) {
    HashMap<Double, Integer> dict = new HashMap<>();
    
    // count
    for (double n : array) {
      if (dict.containsKey(n)) {
        dict.put(n, dict.get(n) + 1);
      } else {
        dict.put(n, 1);
      }
    }

    // find max
    int max = 0;
    double ans = 0;
    for (HashMap.Entry<Double, Integer> entry : dict.entrySet()) {
      if (entry.getValue() > max) {
        max = entry.getValue();
        ans = entry.getKey();
      }
    }

    return ans;
  }
  
  /**
   * Calculates the statistical range (spread) of the data.
   *
   * @param array the input data
   * @return the difference between the max and min values
   */
  public static double range(double[] array) {
    return max(array) - min(array);
  }

  /**
   * Computes the quantile value for a given probability using an arbitrary data array. PPF (Percent Point Function) is another term for quantiles.
   *
   * @param array the input data array
   * @param p the probability threshold between 0.0 and 1.0
   * @return the calculated quantile value
   */
  public static double PPF(double[] array, double p) {
    General.isProbability(p);

    double[] sortedData = array.clone();
    Arrays.sort(sortedData);

    int index = (int) Math.ceil(p / sortedData.length) - 1;
    return sortedData[Math.max(0, index)];
  }

  /**
   * Computes the Cumulative Distribution Function value for an arbitrary array.
   *
   * @param array the input data array
   * @param val the upper limit threshold
   * @return the proportion of values less than or equal to val
   */
  public static double CDF(double[] array, double val) {
    double[] sortedData = array.clone();
    Arrays.sort(sortedData);

    int count = 0;
    for (double x : sortedData) {
      if (x <= val) {
        count++;
      } else {
        break;
      }
    }

    return (double) count / sortedData.length;
  }

  /**
   * Computes the Survival Function value for an arbitrary array.
   *
   * @param array the input data array
   * @param val the lower limit threshold
   * @return the proportion of values greater than or equal to val
   */
  public static double SF(double[] array, double val) {
    return 1 - CDF(array, val);
  }

  /**
   * Computes the probability of data falling within an interval for an arbitrary array.
   *
   * @param array the input data array
   * @param low the exclusive or inclusive lower bound
   * @param upper the exclusive or inclusive upper bound
   * @return the calculated interval probability
   */
  public static double P(double[] array, double low, double upper) {
    return CDF(array, upper) - CDF(array, low);
  }
}
