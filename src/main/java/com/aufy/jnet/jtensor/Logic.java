package com.aufy.jnet.jtensor;

import com.aufy.jnet.core.backend.interfaces.atomic.BinaryOp;
import com.aufy.jnet.core.backend.interfaces.atomic.UnaryOp;
import com.aufy.jnet.core.backend.interfaces.composite.Mapping;
import com.aufy.jnet.core.backend.interfaces.logical.BinaryLogic;
import com.aufy.jnet.core.backend.interfaces.logical.UnaryLogic;

public class Logic {
  public static double accuracy = 0.001;
  private final Tensor tensor;

  public Logic(Tensor tensor) {
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
   * Compares this tensor with another tensor for equality elementwise into a new tensor. 
   * Broadcasting rules apply.
   *
   * @param other the tensor to compare with this tensor.
   * @return a new tensor of the same sizes with 1s for equal elements and 0s for non-equal elements.
   */
  public Tensor equal(Tensor other) {
    return equal(tensor, other);
  }

  /**
   * Compares two tensors for equality elementwise into a new tensor of the same size. 
   * Broadcasting rules apply.
   *
   * @param tensorA the first tensor.
   * @param tensorB the second tensor.
   * @return a new tensor of the same sizes with 1s for equal elements and 0s for non-equal elements.
   */
  public static Tensor equal(Tensor tensorA, Tensor tensorB) {
    return tensorA.ops.elementwise(tensorB, (a,b) -> {return Math.abs(a - b) < accuracy ? 1 : 0;}, BinaryOp::one, BinaryOp::one);
  }

  /**
   * Compares this tensor with another tensor for inequality elementwise into a new tensor. 
   * Broadcasting rules apply.
   *
   * @param other the tensor to compare with this tensor.
   * @return a new tensor of the same sizes with 1s for non-equal elements and 0s for equal elements.
   */
  public Tensor notEqual(Tensor other) {
    return notEqual(tensor, other);
  }

  /**
   * Compares two tensors for inequality elementwise into a new tensor of the same size. 
   * Broadcasting rules apply.
   *
   * @param tensorA the first tensor.
   * @param tensorB the second tensor.
   * @return a new tensor of the same sizes with 1s for non-equal elements and 0s for equal elements.
   */
  public static Tensor notEqual(Tensor tensorA, Tensor tensorB) {
    return tensorA.ops.elementwise(tensorB, (a,b) -> {return Math.abs(a - b) < accuracy ? 0 : 1;}, BinaryOp::one, BinaryOp::one);
  }

  /**
   * Compares this tensor with another tensor elementwise to check if this tensor is greater. 
   * Broadcasting rules apply.
   *
   * @param other the tensor to compare against this tensor.
   * @return a new tensor of the same sizes with 1s where this tensor > other and 0s otherwise.
   */
  public Tensor greater(Tensor other) {
    return greater(tensor, other);
  }

  /**
   * Compares two tensors elementwise to check if the first is greater than the second. 
   * Broadcasting rules apply.
   *
   * @param tensorA the first tensor.
   * @param tensorB the second tensor.
   * @return a new tensor of the same sizes with 1s where tensorA > tensorB and 0s otherwise.
   */
  public static Tensor greater(Tensor tensorA, Tensor tensorB) {
    return tensorA.ops.elementwise(tensorB, (a,b) -> {return a > b ? 1 : 0;}, BinaryOp::one, BinaryOp::one);
  }

  /**
   * Compares this tensor with another tensor elementwise to check if this tensor is lesser. 
   * Broadcasting rules apply.
   *
   * @param other the tensor to compare against this tensor.
   * @return a new tensor of the same sizes with 1s where this tensor < other and 0s otherwise.
   */
  public Tensor lesser(Tensor other) {
    return lesser(tensor, other);
  }

  /**
   * Compares two tensors elementwise to check if the first is less than the second. 
   * Broadcasting rules apply.
   *
   * @param tensorA the first tensor.
   * @param tensorB the second tensor.
   * @return a new tensor of the same sizes with 1s where tensorA < tensorB and 0s otherwise.
   */
  public static Tensor lesser(Tensor tensorA, Tensor tensorB) {
    return tensorA.ops.elementwise(tensorB, (a,b) -> {return a < b ? 1 : 0;}, BinaryOp::one, BinaryOp::one);
  }

  /**
   * Evaluates a unary logic operator elementwise on this tensor and returns specific values depending on the result.
   *
   * @param operator  the unary condition to evaluate.
   * @param trueCase  the value to assign if the condition is true.
   * @param falseCase the value to assign if the condition is false.
   * @return a new tensor containing trueCase or falseCase values based on the condition evaluation.
   */
  public Tensor where(UnaryLogic operator, double trueCase, double falseCase) {
    return where(tensor, operator, trueCase, falseCase);
  }

  /**
   * Evaluates a unary logic operator elementwise on a tensor and returns specific values depending on the result.
   *
   * @param tensorA   the input tensor.
   * @param operator  the unary condition to evaluate.
   * @param trueCase  the value to assign if the condition is true.
   * @param falseCase the value to assign if the condition is false.
   * @return a new tensor containing trueCase or falseCase values based on the condition evaluation.
   */
  public static Tensor where(Tensor tensorA, UnaryLogic operator, double trueCase, double falseCase) {
    return tensorA.ops.elementwise((a) -> {return operator._apply(a) ? trueCase : falseCase;}, UnaryOp::one);
  }

  // BINARY LOGIC ------------------------------------------------------------

  /**
   * Selects elements elementwise between this tensor and another tensor based on a binary logic condition.
   *
   * @param other    the other source tensor.
   * @param operator the binary condition used to make the selection.
   * @return a new tensor matching elements from this tensor where the condition is met, and the other tensor otherwise.
   */
  public Tensor select(Tensor other, BinaryLogic operator) {
    return select(tensor, other, operator);
  }

  /**
   * Selects elements elementwise from two tensors based on a binary logic condition.
   *
   * @param tensorA  the first source tensor.
   * @param tensorB  the second source tensor.
   * @param operator the binary condition used to make the selection.
   * @return a new tensor matching elements from tensorA where the condition is met, and tensorB otherwise.
   */
  public static Tensor select(Tensor tensorA, Tensor tensorB, BinaryLogic operator) {
    return tensorA.ops.elementwise(tensorB, (a,b) -> {return operator._apply(a, b) ? a : b;}, BinaryOp::one, BinaryOp::one);
  }

  // REDUCTION LOGIC ---------------------------------------------------------

  /**
   * Checks if all elements of this tensor along the specified axes satisfy the unary logic operator.
   *
   * @param operator the condition to check against elements.
   * @param axes     the dimensions along which reduction is performed.
   * @return a reduced tensor with 1s where all elements in the slice pass the condition, and 0s otherwise.
   */
  public Tensor all(UnaryLogic operator, int... axes) {
    return all(tensor, operator, axes);
  }

  /**
   * Checks if all elements along the specified axes satisfy the unary logic operator.
   *
   * @param tensorA  the input tensor.
   * @param operator the condition to check against elements.
   * @param axes     the dimensions along which reduction is performed.
   * @return a reduced tensor with 1s where all elements in the slice pass the condition, and 0s otherwise.
   */
  public static Tensor all(Tensor tensorA, UnaryLogic operator, int... axes) {
    return tensorA.ops.reduce(
      (double[] a) -> {
        for (double b : a) if (!operator._apply(b)) return 0;
        return 1;
      }, 
    Mapping::one, axes);
  }

  /**
   * Checks if any element of this tensor along the specified axes satisfies the unary logic operator.
   *
   * @param operator the condition to check against elements.
   * @param axes     the dimensions along which reduction is performed.
   * @return a reduced tensor with 1s where at least one element in the slice passes the condition, and 0s otherwise.
   */
  public Tensor any(UnaryLogic operator, int... axes) {
    return any(tensor, operator, axes);
  }

  /**
   * Checks if any element along the specified axes satisfies the unary logic operator.
   *
   * @param tensorA  the input tensor.
   * @param operator the condition to check against elements.
   * @param axes     the dimensions along which reduction is performed.
   * @return a reduced tensor with 1s where at least one element in the slice passes the condition, and 0s otherwise.
   */
  public static Tensor any(Tensor tensorA, UnaryLogic operator, int... axes) {
    return tensorA.ops.reduce(
      (double[] a) -> {
        for (double b : a) if (operator._apply(b)) return 1;
        return 0;
      }, 
    Mapping::one, axes);
  }


}