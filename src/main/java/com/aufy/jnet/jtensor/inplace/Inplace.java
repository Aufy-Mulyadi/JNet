package com.aufy.jnet.jtensor.inplace;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.arrayops.Tools;
import com.aufy.jnet.core.backend.exceptions.Message;
import com.aufy.jnet.core.backend.exceptions.internal.Data;
import com.aufy.jnet.core.backend.exceptions.tensors.Argument;
import com.aufy.jnet.core.backend.exceptions.tensors.Axes;
import com.aufy.jnet.core.tensor.core.backend.compute.Engine;
import com.aufy.jnet.core.tensor.core.backend.compute.Memory;
import com.aufy.jnet.core.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.jtensor.Internal;
import com.aufy.jnet.jtensor.Tensor;

public class Inplace {
  private final Tensor tensor;
  private final Internal internal;
  
  public Inplace(Tensor tensor) {
    this.tensor = tensor;
    this.internal = new Internal(tensor);
  }

  @Override
  public String toString() {
    if (tensor == null) {
      return "Null";
    } else {
      return tensor.toString();
    }
  }

  // ###########################################################################################################
  // IMPLEMENTATIONS
  // ###########################################################################################################
  
  // BINARY OPERATIONS ---------------------------------

  /**
   * Adds another tensor to this tensor elementwise.
   *
   * @param tensorB the tensor to add.
   * @return a reference to this tensor.
   */
  public Tensor add(Tensor tensorB) {
    add(tensor, tensorB);
    return tensor;
  }

  /**
   * Adds two tensors elementwise, mutating tensorA.
   *
   * @param tensorA the first tensor (mutated).
   * @param tensorB the second tensor.
   * @return a reference to the tensor.
   */
  public static Tensor add(Tensor tensorA, Tensor tensorB) {
    Ops.elementwise(tensorA, tensorB, (x, y) -> x + y);
    return tensorA;
  }

  /**
   * Subtracts another tensor from this tensor elementwise.
   *
   * @param tensorB the tensor to subtract.
   * @return a reference to this tensor.
   */
  public Tensor sub(Tensor tensorB) {
    sub(tensor, tensorB);
    return tensor;
  }

  /**
   * Subtracts one tensor from another elementwise, mutating tensorA.
   *
   * @param tensorA the base tensor (mutated).
   * @param tensorB the tensor to subtract.
   * @return a reference to the tensor.
   */
  public static Tensor sub(Tensor tensorA, Tensor tensorB) {
    Ops.elementwise(tensorA, tensorB, (x, y) -> x - y);
    return tensorA;
  }

  /**
   * Computes the Hadamard product (elementwise multiplication).
   *
   * @param tensorB the tensor to multiply by.
   * @return a reference to this tensor.
   */
  public Tensor hadamard(Tensor tensorB) {
    hadamard(tensor, tensorB);
    return tensor;
  }

  /**
   * Computes the Hadamard product elementwise, mutating tensorA.
   *
   * @param tensorA the first tensor (mutated).
   * @param tensorB the second tensor.
   * @return a reference to the tensor.
   */
  public static Tensor hadamard(Tensor tensorA, Tensor tensorB) {
    Ops.elementwise(tensorA, tensorB, (x, y) -> x * y);
    return tensorA;
  }

  /**
   * Divides this tensor by another tensor elementwise.
   *
   * @param tensorB the denominator tensor.
   * @return a reference to this tensor.
   */
  public Tensor div(Tensor tensorB) {
    div(tensor, tensorB);
    return tensor;
  }

  /**
   * Divides one tensor by another elementwise, mutating tensorA.
   *
   * @param tensorA the numerator tensor (mutated).
   * @param tensorB the denominator tensor.
   * @return a reference to the tensor.
   */
  public static Tensor div(Tensor tensorA, Tensor tensorB) {
    Ops.elementwise(tensorA, tensorB, (x, y) -> x / y);
    return tensorA;
  }

  /**
   * Raises this tensor to the power of another tensor elementwise.
   *
   * @param tensorB the exponent tensor.
   * @return a reference to this tensor.
   */
  public Tensor pow(Tensor tensorB) {
    pow(tensor, tensorB);
    return tensor;
  }

  /**
   * Raises a base tensor to the power of an exponent tensor, mutating tensorA.
   *
   * @param tensorA the base tensor (mutated).
   * @param tensorB the exponent tensor.
   * @return a reference to the tensor.
   */
  public static Tensor pow(Tensor tensorA, Tensor tensorB) {
    Ops.elementwise(tensorA, tensorB, Math::pow);
    return tensorA;
  }

  // GEOMETRY STUFF ---------------------------------

  /**
   * Reshapes this tensor into the specified shape.
   * 
   * @param shape the desired dimensions for the new tensor.
   * @return a reference to this tensor.
   */
  public Tensor reshape(int... shape) {
    reshape(this.tensor, shape);
    return tensor;
  }

  /**
   * Reshapes a specified tensor into a new target shape layout.
   * 
   * @param tensor the tensor to reshape.
   * @param shape the desired dimensions for the new tensor.
   * @return a reference to the tensor.
   */
  public static Tensor reshape(Tensor tensor, int... shape) {
    new Internal(tensor).setShape(Shaping.inferShape(tensor.getShape(), shape));
    return tensor;
  }

  /**
   * Permute the axes order of this tensor according to the specified layout.
   * 
   * @param axes the target ordering index map for the dimensions.
   * @return a reference to this tensor.
   */
  public Tensor permute(int... axes) {
    permute(this.tensor, axes);
    return tensor;
  }

  /**
   * Permute the axes order of this tensor according to the specified layout.
   * 
   * @param tensor the tensor to permute. 
   * @param axes the target ordering index map for the dimensions.
   * @return a reference to the tensor.
   */
  public static Tensor permute(Tensor tensor, int... newOrder) {
    Internal wrapper = new Internal(tensor);

    int[] newShape = Shaping.reorder(tensor.getShape(), newOrder);
    int[] newStrides = Shaping.reorder(wrapper.strides, newOrder);

    wrapper.setData(Engine.makeContiguous(wrapper.data, newShape, newStrides));
    return tensor;
  }

  /**
   * Squeezes single-dimensional configurations (singletons) out of a specified tensor's dimensions.
   * 
   * @param axes the location of singleton dimensions to remove.
   * @return a reference to this tensor.
   */
  public Tensor squeeze(int... axes) {
    squeeze(this.tensor, axes);
    return tensor;
  }

  /**
   * Squeezes single-dimensional configurations (singletons) out of a specified tensor's dimensions.
   * 
   * @param tensor the tensor to squeeze.
   * @param axes the location of singleton dimensions to remove.
   * @return a reference to the tensor.
   */
  public static Tensor squeeze(Tensor tensor, int... axes) {
    if (Reductions.prod(Tools.gather(tensor.getShape(), axes)) != 1) Message.crash("shaping", "cannot squeeze non-singleton dimension(s)", "selective squeezing", "tried to squeeze a non-singleton dimension (dimensions with size 1)");
    Axes.verifyAxes("squeezing", tensor.getRank(), axes);
    
    int[] newShape = Shaping.removeAxes(tensor.getShape(), axes);
    Internal.setShape(tensor, newShape);
    Internal.setShape(tensor, Memory.calculateStrides(newShape));
    return tensor;
  }

  /**
   * Squeezes all single-dimensional configurations (singletons) out of this tensor's dimensions.
   * @return a reference to this tensor.
   */
  public Tensor squeeze() {
    squeeze(this.tensor);
    return tensor;
  }

  /**
   * Squeezes all single-dimensional configurations (singletons) out of this tensor's dimensions.
   * 
   * @param tensor the tensor to squeeze.
   * @return a reference to the tensor.
   */
  public static Tensor squeeze(Tensor tensor) {
    int[] newShape = Shaping.removeAxes(tensor.getShape(), Tools.findIndices(tensor.getShape(), 1));
    Internal.setShape(tensor, newShape);
    Internal.setShape(tensor, Memory.calculateStrides(newShape));
    return tensor;
  }

  /**
   * Unsqueezes this tensor's dimensions at the specified axes.
   * 
   * @param axes the location of the new dimensions to expand.
   * @return a reference to this tensor.
   */
  public Tensor unsqueeze(int... axes) {
    unsqueeze(this.tensor, axes);
    return tensor;
  }

  /**
   * Unsqueezes a single dimension into a specified tensor's structural layout.
   * 
   * @param tensor the tensor to unsqueeze.
   * @param axes the location of the new dimensions to expand.
   * @return a reference to the tensor.
   */
  public static Tensor unsqueeze(Tensor tensor, int... axes) {
    Axes.verifyUniqueList("unsqueeze", axes);

    int[] newShape = Shaping.unsqueeze(tensor.getShape(), axes);
    Internal.setShape(tensor, newShape);
    Internal.setShape(tensor, Memory.calculateStrides(newShape));
    return tensor;
  }

  /**
   * Flattens this tensor into a single-dimensional layout.
   * @return a reference to this tensor.
   */
  public Tensor flatten() {
    flatten(tensor);
    return tensor;
  }

  /**
   * Flattens a tensor into a single-dimensional layout.
   *
   * @param tensor the tensor to flatten.
   * @return a reference to the tensor.
   */
  public static Tensor flatten(Tensor tensor) {
    reshape(tensor, -1);
    return tensor;
  }

  /**
   * Transposes the dimensional axes of this tensor.
   * @return a reference to this tensor.
   */
  public Tensor transpose() {
    transpose(tensor);
    return tensor;
  }

  /**
   * Transposes the dimensional axes of a tensor. In this implementation, 
   *
   * @param tensor the tensor to transpose.
   * @return a reference to the tensor.
   */
  public static Tensor transpose(Tensor tensor) {
    if (tensor.getRank() < 2) Message.crash("mathematical", "transposing a vector or scalar", "tensor transpose", "the transpose of a vector or scalar is undefined");

    int[] transposed = Tools.arrange(tensor.getRank());

    // swapping the last 2 axes
    int temp = transposed[tensor.getRank() - 1];
    transposed[tensor.getRank() - 1] = transposed[tensor.getRank() - 2];
    transposed[tensor.getRank() - 2] = temp;

    permute(tensor, transposed);

    return tensor;
  }
  
  // UNARY OPERATIONS ---------------------------------

  /**
   * Adds a scalar value to every element in this tensor.
   *
   * @param scalar the value to add.
   * @return a reference to this tensor.
   */
  public Tensor add(double scalar) {
    add(tensor, scalar);
    return tensor;
  }

  /**
   * Adds a scalar value to every element in a tensor.
   *
   * @param tensor the tensor to modify.
   * @param scalar the value to add.
   * @return a reference to the tensor.
   */
  public static Tensor add(Tensor tensor, double scalar) {
    Ops.elementwise(tensor, (a) -> a + scalar);
    return tensor;
  }

  /**
   * Subtracts a scalar value to every element in this tensor.
   *
   * @param scalar the value to subtract.
   * @return a new shifted tensor.
   * @return a reference to this tensor.
   */
  public Tensor sub(double scalar) {
    sub(tensor, scalar);
    return tensor;
  }

  /**
   * Subtracts a scalar value to every element in a tensor.
   *
   * @param tensor the tensor to modify.
   * @param scalar the value to subtract.
   * @return a new shifted tensor.
   * @return a reference to the tensor.
   */
  public static Tensor sub(Tensor tensor, double scalar) {
    Ops.elementwise(tensor, (a) -> a - scalar);
    return tensor;
  }

  /**
   * Multiplies every element in this tensor by a scalar value.
   *
   * @param scalar the scaling factor.
   * @return a reference to this tensor.
   */
  public Tensor mul(double scalar) {
    mul(tensor, scalar);
    return tensor;
  }

  /**
   * Multiplies every element in a tensor by a scalar value.
   *
   * @param tensor the tensor to scale.
   * @param scalar the scaling factor.
   * @return a reference to the tensor.
   */
  public static Tensor mul(Tensor tensor, double scalar) {
    Ops.elementwise(tensor, (a) -> a * scalar);
    return tensor;
  }

  /**
   * Divides every element in this tensor by a scalar value.
   *
   * @param scalar the scaling factor.
   * @return a new scaled tensor.
   * @return a reference to this tensor.
   */
  public Tensor div(double scalar) {
    div(tensor, scalar);
    return tensor;
  }

  /**
   * Divides every element in a tensor by a scalar value.
   *
   * @param tensor the tensor to scale.
   * @param scalar the dividing factor.
   * @return a new scaled tensor.
   * @return a reference to the tensor.
   */
  public static Tensor div(Tensor tensor, double scalar) {
    Ops.elementwise(tensor, (a) -> a / scalar);
    return tensor;
  }

  /**
   * Raises every element in this tensor to a scalar power.
   *
   * @param power the exponent value.
   * @return a reference to this tensor.
   */
  public Tensor pow(double power) {
    pow(tensor, power);
    return tensor;
  }

  /**
   * Raises every element in a tensor to a scalar power.
   *
   * @param tensor the base tensor.
   * @param scalar the exponent value.
   * @return a reference to the tensor.
   */
  public static Tensor pow(Tensor tensor, double power) {
    Ops.elementwise(tensor, (a) -> Math.pow(a, power));
    return tensor;
  }

  /**
   * Exponentiates every element in this tensor to a scalar power (scalar^element).
   *
   * @param base the base value.
   * @return a reference to this tensor.
   */
  public Tensor exp(double base) {
    exp(tensor, base);
    return tensor;
  }

  /**
   * Exponentiates every element in this tensor to a scalar power (scalar^element).
   *
   * @param tensor the exponent tensor.
   * @param base the base value.
   * @return a reference to the tensor.
   */
  public static Tensor exp(Tensor tensor, double base) {
    Argument.aboveValue(0, base, "exp");
    Ops.elementwise(tensor, (a) -> Math.pow(base, a));
    return tensor;
  }

  /**
   * Exponentiates every element in this tensor.
   * @return a reference to this tensor.
   */
  public Tensor exp() {
    exp(tensor);
    return tensor;
  }

  /**
   * Exponentiates every element in this tensor.
   *
   * @param tensor the base tensor.
   * @return a reference to the tensor.
   */
  public static Tensor exp(Tensor tensor) {
    Ops.elementwise(tensor, (a) -> Math.exp(a));
    return tensor;
  }

  /**
   * Take the logarithm of every element in this tensor using a specified base.
   * @return a reference to this tensor.
   */
  public Tensor log(double base) {
    log(tensor, base);
    return tensor;
  }

  /**
   * Exponentiates every element in this tensor using a specified base.
   *
   * @param tensor the base tensor.
   * @param base the base of the logarithm.
   * @return a reference to the tensor.
   */
  public static Tensor log(Tensor tensor, double base) {
    Data.aboveValue(0, tensor.dump(), "log");
    Argument.aboveValue(0, base, "log");
    Ops.elementwise(tensor, (a) -> Math.log(a)/Math.log(base));
    return tensor;
  }

  /**
   * Take the logarithm of every element in this tensor using a specified base.
   * @return a reference to this tensor.
   */
  public Tensor ln() {
    ln(tensor);
    return tensor;
  }

  /**
   * Exponentiates every element in this tensor using a specified base.
   *
   * @param tensor the base tensor.
   * @param base the base of the logarithm.
   * @return a reference to the tensor.
   */
  public static Tensor ln(Tensor tensor) {
    Data.aboveValue(0, tensor.dump(), "ln (natural log)");

    Ops.elementwise(tensor, (a) -> Math.log(a));
    return tensor;
  }

  /**
   * Takes the square root of every element in this tensor.
   *
   * @param tensor the base tensor.
   * @param root the root value.
   * @return a new exponentiated tensor.
   */
  public Tensor root() {
    root(tensor);
    return tensor;
  }

  /**
   * Takes the square root of every element in this tensor.
   *
   * @param tensor the base tensor.
   * @param root the root value.
   * @return a new exponentiated tensor.
   */
  public static Tensor root(Tensor tensor) {
    Ops.elementwise(tensor, (a) -> Math.pow(a, 0.5));
    return tensor;
  }

  /**
   * Takes the root of every element in this tensor.
   *
   * @param tensor the base tensor.
   * @param root the root value.
   * @return a new exponentiated tensor.
   */
  public Tensor root(double root) {
    return root(tensor, root);
  }

  /**
   * Takes the root of every element in this tensor.
   *
   * @param tensor the base tensor.
   * @param root the root value.
   * @return a new exponentiated tensor.
   */
  public static Tensor root(Tensor tensor, double root) {
    Ops.elementwise(tensor, (a) -> Math.pow(a, 1.0 / root));
    return tensor;
  }
}
