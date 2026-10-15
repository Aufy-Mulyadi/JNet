package com.aufy.jnet.jtensor;

import com.aufy.jnet.core.backend.arrayops.Moments;
import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.arrayops.Tools;
import com.aufy.jnet.core.backend.arrayops.Uniform;
import com.aufy.jnet.core.backend.exceptions.statistics.General;
import com.aufy.jnet.core.backend.exceptions.statistics.Parameter;
import com.aufy.jnet.core.backend.interfaces.atomic.AccumulatorOp;
import com.aufy.jnet.core.backend.interfaces.atomic.UnaryOp;

public class Stats {
  private final Tensor tensor;
  private final int[] axes;

  public Stats(Tensor tensor, int... axes) {
    this.tensor = tensor;
    this.axes = axes;
  }

  @Override
  public String toString() {
    if (tensor == null) {
      return "UninitializedTensor[]()";
    } else {
      return "Reference" + tensor;
    }
  }

  // ###########################################################################################################
  // descriptive
  // ###########################################################################################################

  /**
   * Computes the mean of the elements of this tensor along the specified axes.
   * 
   * @param tensor the tensor to reduce.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the means.
   */
  public Tensor mean() {
    return mean(tensor, axes);
  }

  /**
   * Computes the mean of the elements of a tensor along the specified axes.
   * 
   * @param tensor the tensor to reduce.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the means.
   */
  public static Tensor mean(Tensor tensor, int... axes) {
    return Ops.reduce(tensor, Moments::mean, 
      (double[] input) -> {
        return Uniform.add(input, 1.0 / input.length);
      }, axes
    );
  }

  /**
   * Computes the variance of the elements of this tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the variances.
   */
  public Tensor variance() {
    return variance(tensor, axes);
  }

  /**
   * Computes the variance of the elements of a tensor along the specified axes.
   *
   * @param tensor the tensor to reduce.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the variances.
   */
  public static Tensor variance(Tensor tensor, int... axes) {
    return Ops.reduce(tensor, Moments::variance,
      (double[] input) -> {
        double[] out = new double[input.length];
        double mean = Moments.mean(input);

        for (int i = 0; i < input.length; i++) out[i] = 2.0 * (input[i] - mean) / input.length;

        return out;
      },
      axes
    );
  }

  /**
   * Computes the skewness of the elements of this tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the skewness values.
   */
  public Tensor skew() {
    return skew(tensor, axes);
  }

  /**
   * Computes the skewness of the elements of a tensor along the specified axes.
   *
   * @param tensor the tensor to reduce.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the skewness values.
   */
  public static Tensor skew(Tensor tensor, int... axes) {
    return Ops.reduce(tensor, Moments::skew,
      (double[] input) -> {
        double[] out = new double[input.length];

        double mean = Moments.mean(input);
        double[] shifted = Uniform.subtract(input, -mean);

        double m2 = AccumulatorOp.apply(shifted, 0.0, (a, b) -> a + Math.pow(b, 2)) / input.length;
        double m3 = AccumulatorOp.apply(shifted, 0.0, (a, b) -> a + Math.pow(b, 3)) / input.length;

        double stdev = Math.sqrt(m2/input.length);

        for (int i = 0; i < input.length; i++) {
          out[i] =
            (3.0 * Math.pow(shifted[i], 2)) / (input.length * Math.pow(stdev, 3))
            -
            (3.0 * m3 * shifted[i]) / (input.length * Math.pow(stdev, 5));
        }

        return out;
      },
      axes
    );
  }

  /**
   * Computes the kurtosis of the elements of this tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the kurtosis values.
   */
  public Tensor kurtosis() {
    return kurtosis(tensor, axes);
  }

  /**
   * Computes the kurtosis of the elements of a tensor along the specified axes.
   *
   * @param tensor the tensor to reduce.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the kurtosis values.
   */
  public static Tensor kurtosis(Tensor tensor, int... axes) {
    return Ops.reduce(tensor, Moments::kurtosis,
      (double[] input) -> {
        double[] out = new double[input.length];
        double mean = Moments.mean(input);
        double[] shifted = Uniform.subtract(input, -mean);

        double m2 = AccumulatorOp.apply(shifted, 0.0, (a, b) -> a + Math.pow(b, 2)) / input.length;
        double m4 = AccumulatorOp.apply(shifted, 0.0, (a, b) -> a + Math.pow(b, 4)) / input.length;

        for (int i = 0; i < input.length; i++) {
          out[i] =
            (4.0 * Math.pow(shifted[i], 3)) / (input.length * Math.pow(m2, 2))
            -
            (4.0 * m4 * shifted[i]) / (input.length * Math.pow(m2, 3));
        }

        return out;
      },
      axes
    );
  }

  // ###########################################################################################################
  // empirical
  // ###########################################################################################################
  
  /**
   * Computes the quantile value for a given probability in selected axes of a tensor. 
   * PPF (Percent Point Function) is another term for quantiles.
   *
   * @param array the input Tensor.
   * @param p the probability threshold between 0.0 and 1.0 inclusive.
   * @return the calculated quantile value.
   */
  public Tensor PPF(double p) {
    General.isProbability(p);

    return PPF(tensor, p, axes);
  }

  /**
   * Computes the quantile value for a given probability in selected axes of a tensor. 
   * PPF (Percent Point Function) is another term for quantiles.
   *
   * @param array the input Tensor.
   * @param p the probability threshold between 0.0 and 1.0 inclusive.
   * @return the calculated quantile value.
   */
  public static Tensor PPF(Tensor tensor, double p, int... axes) {
    General.isProbability(p);

    return Ops.reduce(tensor, 
    (double[] input) -> {
      return com.aufy.jnet.core.backend.arrayops.Statistics.PPF(input, p);
    },
    Tools::onesLike,
    axes);
  }

  /**
   * Computes the Cumulative Distribution Function value in selected axes of a tensor.
   *
   * @param tensor the input data tensor.
   * @param val the upper limit threshold.
   * @return the proportion of values less than or equal to val.
   */
  public Tensor CDF(double val) {
    return CDF(tensor, val, axes);
  }

  /**
   * Computes the Cumulative Distribution Function value in selected axes of a tensor.
   *
   * @param tensor the input data tensor.
   * @param val the upper limit threshold.
   * @return the proportion of values less than or equal to val.
   */
  public static Tensor CDF(Tensor tensor, double val, int... axes) {
    return Ops.reduce(tensor, 
    (double[] input) -> {
      return com.aufy.jnet.core.backend.arrayops.Statistics.CDF(input, val);
    },
    Tools::onesLike,
    axes);
  }

  /**
   * Computes the Survival Function value in selected axes of a tensor.
   *
   * @param tensor the input data tensor.
   * @param val the lower limit threshold.
   * @return the proportion of values greater than or equal to val.
   */
  public Tensor SF(double val) {
    return SF(tensor, val, axes);
  }

  /**
   * Computes the Survival Function value in selected axes of a tensor.
   *
   * @param tensor the input data array
   * @param val the lower limit threshold
   * @return the proportion of values greater than or equal to val
   */
  public static Tensor SF(Tensor tensor, double val, int... axes) {
    return Ops.reduce(tensor, 
    (double[] input) -> {
      return com.aufy.jnet.core.backend.arrayops.Statistics.SF(input, val);
    },
    Tools::onesLike,
    axes);
  }

  /**
   * Computes the probability of data falling within an interval in selected axes of a tensor.
   *
   * @param low the exclusive or inclusive lower bound.
   * @param upper the exclusive or inclusive upper bound.
   * @return the calculated interval probability.
   */
  public Tensor P(double low, double upper) {
    Parameter.isAboveValue(upper, low, "P (Probability Function)");
    
    return P(tensor, low, upper, axes);
  }

  /**
   * Computes the probability of data falling within an interval in selected axes of a tensor.
   *
   * @param tensor the input data tensor.
   * @param low the exclusive or inclusive lower bound.
   * @param upper the exclusive or inclusive upper bound.
   * @return the calculated interval probability.
   */
  public static Tensor P(Tensor tensor, double low, double upper, int... axes) {
    Parameter.isAboveValue(upper, low, "P (Probability Function)");

    return Ops.reduce(tensor, 
    (double[] input) -> {
      return com.aufy.jnet.core.backend.arrayops.Statistics.P(input, low, upper);
    },
    Tools::onesLike,
    axes);
  }
  
  // ###########################################################################################################
  // other
  // ###########################################################################################################

  /**
   * Computes the vector norm in selected axes of this tensor. Use max() for infinity norm for stability.
   * 
   * @param n The norm.
   * @param axes The axes to reduce.
   * @return The vector norm across axes.
   */
  public Tensor norm(double n) {
    return Ops.reduce(tensor,
    (double[] input) -> {
      return Math.pow(Reductions.sum(Uniform.pow(input, n)), 1.0 / n);
    },
    input -> UnaryOp.apply(input, a -> n * Math.pow(a, n - 1)),
    axes);
  }

  /**
   * Computes the vector norm in selected axes of a tensor. Use max() for infinity norm for stability.
   * 
   * @param tensor The tensor to measure.
   * @param n The norm.
   * @param axes The axes to reduce.
   * @return The vector norm across axes.
   */
  public static Tensor norm(Tensor tensor, double n, int... axes) {

    return Ops.reduce(tensor, 
    (double[] input) -> {
      return Math.pow(Reductions.sum(Uniform.pow(input, n)), 1.0 / n);
    },
    input -> UnaryOp.apply(input, a -> n * Math.pow(a, n - 1)),
    axes);
  }

}
