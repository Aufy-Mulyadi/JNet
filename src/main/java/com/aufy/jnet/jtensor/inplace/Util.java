package com.aufy.jnet.jtensor.inplace;

import com.aufy.jnet.core.backend.exceptions.tensors.Argument;
import com.aufy.jnet.core.backend.scalarops.Miscellaneous;
import com.aufy.jnet.jtensor.Tensor;

public class Util {
  private final Tensor tensor;

  public Util(Tensor tensor) {
    this.tensor = tensor;
  }
  
  @Override
  public String toString() {
    if (tensor == null) {
      return "Null";
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
   * @return a reference to this tensor.
   */
  public Tensor abs() {
    abs(tensor);
    return tensor;
  }

  /**
   * Returns the absolute value of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @return a reference to the tensor.
   */
  public static Tensor abs(Tensor tensor) {
    Ops.elementwise(tensor, Math::abs);
    return tensor;
  }

  /**
   * Returns the sign of each element in this tensor.
   *
   * @return a reference to this tensor.
   */
  public Tensor sign() {
    sign(tensor);
    return tensor;
  }

  /**
   * Returns the sign of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @return a reference to the tensor.
   */
  public static Tensor sign(Tensor tensor) {
    Ops.elementwise(tensor, Math::signum);
    return tensor;
  }

  /**
   * Returns the ceiling of each element in this tensor.
   *
   * @return a reference to this tensor.
   */
  public Tensor ceil() {
    ceil(tensor);
    return tensor;
  }

  /**
   * Returns the ceiling of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @return a reference to the tensor.
   */
  public static Tensor ceil(Tensor tensor) {
    Ops.elementwise(tensor, Math::ceil);
    return tensor;
  }

  /**
   * Returns the floor of each element in this tensor.
   *
   * @return a reference to this tensor.
   */
  public Tensor floor() {
    floor(tensor);
    return tensor;
  }

  /**
   * Returns the floor of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @return a reference to the tensor.
   */
  public static Tensor floor(Tensor tensor) {
    Ops.elementwise(tensor, Math::floor);
    return tensor;
  }

  /**
   * Returns the rounded value of each element in this tensor.
   *
   * @param scale the number of decimal places to round to.
   * @return a reference to this tensor.
   */
  public Tensor round(int scale) {
    round(tensor, scale);
    return tensor;
  }

  /**
   * Returns the rounded value of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @param scale the number of decimal places to round to.
   * @return a reference to the tensor.
   */
  public static Tensor round(Tensor tensor, int scale) {
    Argument.belowValue(0, scale, "rounding");
    Ops.elementwise(tensor, a -> Miscellaneous.round(a, scale));
    return tensor;
  }

  /**
   * Returns the truncated value of each element in this tensor.
   *
   * @param scale the number of decimal places to truncate to.
   * @return a reference to this tensor.
   */
  public Tensor trunc(int scale) {
    trunc(tensor, scale);
    return tensor;
  }

  /**
   * Returns the truncated value of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @param scale the number of decimal places to truncate to.
   * @return a reference to the tensor.
   */
  public static Tensor trunc(Tensor tensor, int scale) {
    Argument.belowValue(0, scale, "truncate");
    Ops.elementwise(tensor, a -> Miscellaneous.truncate(a, scale));
    return tensor;
  }

  /**
   * Returns the fractional part of each element in this tensor.
   *
   * @param scale the number of decimal places to keep for the fractional part.
   * @return a reference to this tensor.
   */
  public Tensor frac(int scale) {
    frac(tensor, scale);
    return tensor;
  }

  /**
   * Returns the fractional part of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @param scale the number of decimal places to keep for the fractional part.
   * @return a reference to the tensor.
   */
  public static Tensor frac(Tensor tensor, int scale) {
    Argument.belowValue(0, scale, "fractional");
    Ops.elementwise(tensor, a -> Miscellaneous.fractional(a, scale));
    return tensor;
  }

  /**
   * Returns the clipped value of each element in this tensor.
   *
   * @param min the minimum value to clip to.
   * @param max the maximum value to clip to.
   * @return a reference to this tensor.
   * @return a reference to this tensor.
   */
  public Tensor clip(double min, double max) {
    clip(tensor, min, max);
    return tensor;
  }

  /**
   * Returns the clipped value of each element in this tensor.
   *  
   * @param tensor the tensor to evaluate.
   * @param min the minimum value to clip to.
   * @param max the maximum value to clip to.
   * @return a reference to the tensor.
   */
  public static Tensor clip(Tensor tensor, double min, double max) {
    Argument.aboveValue(min, max, "clipping");
    Argument.belowValue(max, min, "clipping");
    Ops.elementwise(tensor, a -> Math.min(Math.max(a, min), max));

    return tensor;
  }
}
