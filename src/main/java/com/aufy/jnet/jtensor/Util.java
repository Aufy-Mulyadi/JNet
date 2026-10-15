package com.aufy.jnet.jtensor;

import com.aufy.jnet.core.backend.exceptions.tensors.Argument;
import com.aufy.jnet.core.backend.scalarops.Miscellaneous;
import com.aufy.jnet.core.backend.scalarops.Unary;

public class Util {
  private final Tensor tensor;

  public Util(Tensor tensor) {
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
   * Returns the absolute value of each element in this tensor.
   *
   * @return a new tensor containing the absolute values.
   */
  public Tensor abs() {
    return abs(tensor);
  }

  /**
   * Returns the absolute value of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @return a new tensor containing the absolute values.
   */
  public static Tensor abs(Tensor tensor) {
    return Ops.elementwise(tensor, Math::abs, a -> (a > 0) ? 1.0 : -1.0);
  }
  /**
   * Returns the sign of each element in this tensor.
   *
   * @return a new tensor containing the signs.
   */
  public Tensor sign() {
    return sign(tensor);
  }

  /**
   * Returns the sign of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @return a new tensor containing the signs.
   */
  public static Tensor sign(Tensor tensor) {
    return Ops.elementwise(tensor, Math::signum, a -> 0.0);
  }

  /**
   * Returns the ceiling of each element in this tensor.
   *
   * @return a new tensor containing the ceilings.
   */
  public Tensor ceil() {
    return ceil(tensor);
  }

  /**
   * Returns the ceiling of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @return a new tensor containing the ceilings.
   */
  public static Tensor ceil(Tensor tensor) {
    return Ops.elementwise(tensor, Math::ceil, a -> 1.0);
  }

  /**
   * Returns the floor of each element in this tensor.
   *
   * @return a new tensor containing the floors.
   */
  public Tensor floor() {
    return floor(tensor);
  }

  /**
   * Returns the floor of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @return a new tensor containing the floors.
   */
  public static Tensor floor(Tensor tensor) {
    return Ops.elementwise(tensor, Math::floor, a -> 1.0);
  }

  /**
   * Returns the rounded value of each element in this tensor.
   *
   * @param scale the number of decimal places to round to.
   * @return a new tensor containing the rounded values.
   */
  public Tensor round(int scale) {
    return round(tensor, scale);
  }

  /**
   * Returns the rounded value of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @param scale the number of decimal places to round to.
   * @return a new tensor containing the rounded values.
   */
  public static Tensor round(Tensor tensor, int scale) {
    Argument.belowValue(0, scale, "rounding");
    return Ops.elementwise(tensor, a -> Miscellaneous.round(a, scale), a -> 1.0);
  }

  /**
   * Returns the truncated value of each element in this tensor.
   *
   * @param scale the number of decimal places to truncate to.
   * @return a new tensor containing the truncated values.
   */
  public Tensor trunc(int scale) {
    return trunc(tensor, scale);
  }

  /**
   * Returns the truncated value of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @param scale the number of decimal places to truncate to.
   * @return a new tensor containing the truncated values.
   */
  public static Tensor trunc(Tensor tensor, int scale) {
    Argument.belowValue(0, scale, "truncate");
    return Ops.elementwise(tensor, a -> Miscellaneous.truncate(a, scale), a -> 1.0);
  }

  /**
   * Returns the fractional part of each element in this tensor.
   *
   * @param scale the number of decimal places to keep for the fractional part.
   * @return a new tensor containing the fractional parts.
   */
  public Tensor frac(int scale) {
    return frac(tensor, scale);
  }

  /**
   * Returns the fractional part of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @param scale the number of decimal places to keep for the fractional part.
   * @return a new tensor containing the fractional parts.
   */
  public static Tensor frac(Tensor tensor, int scale) {
    Argument.belowValue(0, scale, "fractional");
    return Ops.elementwise(tensor, a -> Miscellaneous.fractional(a, scale), a -> 1.0);
  }

  /**
   * Returns the clipped value of each element in this tensor.
   *
   * @param min the minimum value to clip to.
   * @param max the maximum value to clip to.
   * @return a new tensor containing the clipped values.
   */
  public Tensor clip(double min, double max) {
    return clip(tensor, min, max);
  }

  /**
   * Returns the clipped value of each element in this tensor.
   *  
   * @param tensor the tensor to evaluate.
   * @param min the minimum value to clip to.
   * @param max the maximum value to clip to.
   * @return a new tensor containing the clipped values.
   */
  public static Tensor clip(Tensor tensor, double min, double max) {
    Argument.aboveValue(min, max, "clipping");
    Argument.belowValue(max, min, "clipping");
    return Ops.elementwise(tensor, a -> Math.min(Math.max(a, min), max), a -> Unary.step(a-min) * Unary.step(-(a-max)));
  }
}
