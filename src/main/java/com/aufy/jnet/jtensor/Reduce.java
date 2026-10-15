package com.aufy.jnet.jtensor;

import com.aufy.jnet.core.backend.arrayops.Moments;
import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.arrayops.Tools;
import com.aufy.jnet.core.backend.arrayops.Uniform;
import com.aufy.jnet.core.tensor.core.backend.compute.Generator;

public class Reduce {
  private final Tensor tensor;

  public Reduce(Tensor tensor) {
    this.tensor = tensor;
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
  // IMPLEMENTATIONS
  // ###########################################################################################################

  /**
   * Computes the log sum of exponentials of the elements of a tensor along the specified axes. The "Logsumexp" trick
   * is applied to prevent instability.
   * 
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the log sum of exponentials.
   */
  public Tensor logSumExp(int... axes) {
    return Ops.reduce(tensor, 
      Reductions::logSumExp,
      Uniform::softmax,
      axes
    );
  }

  /**
   * Computes the log sum of exponentials of the elements of a tensor along the specified axes. The "Logsumexp" trick
   * is applied to prevent instability.
   * 
   * @param tensor the tensor to reduce.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the log sum of exponentials.
   */
  public static Tensor logSumExp(Tensor tensor, int... axes) {
    return Ops.reduce(tensor, 
      Reductions::logSumExp,
      Uniform::softmax,
      axes
    );
  }

  /**
   * Computes the root mean square of the elements of a tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the root mean squares.
   */
  public Tensor rootMeanSquare(int... axes) {
    return rootMeanSquare(tensor, axes);
  }

  /**
   * Computes the root mean square of the elements of a tensor along the specified axes.
   *
   * @param tensor the tensor to reduce.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the root mean squares.
   */
  public static Tensor rootMeanSquare(Tensor tensor, int... axes) {
    return Ops.reduce(tensor, Moments::rootMeanSquare,
      (double[] input) -> {
        double[] out = new double[input.length];
        double rms = Moments.rootMeanSquare(input);

        for (int i = 0; i < input.length; i++) out[i] = input[i] / (input.length * rms);

        return out;
      }, 
      axes
    );
  }

  /**
   * Returns the index of the maximum value of the elements of a tensor along the specified axes.
   * 
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the index of the maximum values.
   */
  public Tensor argmax(int... axes) { 
    return argmax(tensor, axes);
  }
    
  /**
   * Returns the index of the maximum value of the elements of a tensor along the specified axes.
   * 
   * @param tensor the tensor to reduce.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the index of the maximum values.
   */
  public static Tensor argmax(Tensor tensor, int... axes) { 
    return Ops.reduce(tensor, Reductions::argmax,
      (double[] input) -> {
        double[] result = new double[input.length];
        int index = (int) Reductions.argmax(input);

        result[index] = 1;
        return result;
      }, axes
    );
  }

  /**
   * Returns the index of the minimum value of the elements of a tensor along the specified axes.
   * 
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the index of the minimum values.
   */
  public Tensor argmin(int... axes) { 
    return argmin(tensor, axes);
  }
    
  /**
   * Returns the index of the minimum value of the elements of a tensor along the specified axes.
   * 
   * @param tensor the tensor to reduce.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the index of the minimum values.
   */
  public static Tensor argmin(Tensor tensor, int... axes) { 
    return Ops.reduce(tensor, Reductions::argmin,
      (double[] input) -> {
        double[] result = new double[input.length];
        int index = (int) Reductions.argmin(input);

        result[index] = 1;
        return result;
      }, axes
    );
  }

  // ################################################################################################
  // statistics
  // ################################################################################################

  /**
   * Sums the elements of this tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the sums.
   */
  public Tensor sum(int... axes) {
    return sum(tensor, axes);
  }

  /**
   * Sums the elements of a tensor along the specified axes.
   *
   * @param tensor the tensor to reduce.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the sums.
   */
  public static Tensor sum(Tensor tensor, int... axes) {
    return Ops.reduce(tensor, Reductions::sum, 
      (double[] input) -> {
        return Uniform.add(input, 1.0);
      }, axes
    );
  }

  /**
   * Computes the product of the elements of this tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the products.
   */
  public Tensor prod(int... axes) {
    return prod(tensor, axes);
  }

  /**
   * Computes the product of the elements of a tensor along the specified axes.
   *
   * @param tensor the tensor to reduce.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the products.
   */
  public static Tensor prod(Tensor tensor, int... axes) {
    return Ops.reduce(tensor, Reductions::prod, 
      (double[] input) -> {
        switch (Tools.countContains(input, 0.0)) {
          case 0 -> {
            double[] result = new double[input.length];
            double prod = Reductions.prod(input);

            for (int i = 0; i < input.length; i++) input[i] = prod/input[i];
            
            return result;
          }
          case 1 -> {
            double[] result = new double[input.length];
            int location = Tools.indexOf(input, 0.0);
            result[location] = Reductions.prod(Tools.remove(input, location));
            
            return result;
          }
          default -> {
            return Generator.zeros(input.length);
          }
        }
      }, axes
    );
  }

  /**
   * Finds the minimum values of this tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the minimums.
   */
  public Tensor min(int... axes) {
    return min(tensor, axes);
  }

  /**
   * Finds the minimum values of a tensor along the specified axes.
   *
   * @param tensor the tensor to evaluate.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the minimums.
   */
  public static Tensor min(Tensor tensor, int... axes) {
    return Ops.reduce(tensor, Reductions::min, 
      (double[] input) -> {
        double[] result = new double[input.length];
        double minimum = Reductions.min(input);

        for (int i = 0; i < input.length; i++) if (input[i] == minimum) input[i] = 1;

        return result;
      }, axes
    );
  }

  /**
   * Finds the maximum values of this tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the maximums.
   */
  public Tensor max(int... axes) {
    return max(tensor, axes);
  }

  /**
   * Finds the maximum values of a tensor along the specified axes.
   *
   * @param tensor the tensor to evaluate.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the maximums.
   */
  public static Tensor max(Tensor tensor, int... axes) {
    return Ops.reduce(tensor, Reductions::max, 
      (double[] input) -> {
        double[] result = new double[input.length];
        double maximum = Reductions.max(input);

        for (int i = 0; i < input.length; i++) if (input[i] == maximum) input[i] = 1;

        return result;
      }, axes
    );
  }

}
