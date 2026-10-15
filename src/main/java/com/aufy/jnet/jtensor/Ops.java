package com.aufy.jnet.jtensor;

import com.aufy.jnet.core.backend.exceptions.Message;
import com.aufy.jnet.core.backend.exceptions.tensors.Axes;
import com.aufy.jnet.core.backend.interfaces.atomic.BinaryOp;
import com.aufy.jnet.core.backend.interfaces.atomic.UnaryOp;
import com.aufy.jnet.core.backend.interfaces.composite.Expansion;
import com.aufy.jnet.core.backend.interfaces.composite.Mapping;
import com.aufy.jnet.core.backend.interfaces.composite.Reduction;
import com.aufy.jnet.core.tensor.graph.main.BinaryOps;
import com.aufy.jnet.core.tensor.graph.main.SubtensorOps;
import com.aufy.jnet.core.tensor.graph.main.UnaryOps;

public class Ops {
  private final Tensor tensor;

  public Ops(Tensor tensor) {
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

  
  // BINARY OPERATIONS ---------------------------------

  /**
   * Performs an elementwise binary operation on this tensor and another tensor.
   *
   * @param tensorB the second tensor involved in the operation.
   * @param forwardOperation the binary function to apply to elements.
   * @param dA the partial derivative function with respect to the first tensor.
   * @param dB the partial derivative function with respect to the second tensor.
   * @return a new tensor containing the results of the elementwise operation.
   */
  public Tensor elementwise(Tensor tensorB, BinaryOp forwardOperation, BinaryOp dA, BinaryOp dB) {
    return elementwise(tensor, tensorB, forwardOperation, dA, dB);
  }

  /**
   * Performs an elementwise binary operation between two specified tensors.
   *
   * @param tensorA the first tensor.
   * @param tensorB the second tensor.
   * @param forwardOperation the binary function to apply to elements.
   * @param dA the partial derivative function with respect to tensorA.
   * @param dB the partial derivative function with respect to tensorB.
   * @return a new tensor containing the results of the elementwise operation.
   */
  public static Tensor elementwise(Tensor tensorA, Tensor tensorB, BinaryOp forwardOperation, BinaryOp dA, BinaryOp dB) {
    if (tensorA.getRank() != tensorB.getRank()) Message.crash("internal", "tensors have differing ranks", "tensor elementwise", "both tensors must have the same rank");
    if (forwardOperation == null) Message.crash("internal", "forward operation is null", "tensor elementwise", "forward operation cannot not be null");
    if (dA == null) Message.crash("internal", "partial derivative of tensor A is null", "tensor elementwise", "partial derivative of tensor A cannot not be null");
    if (dB == null) Message.crash("internal", "partial derivative of tensor B is null", "tensor elementwise", "partial derivative of tensor B cannot not be null");
    
    return new Tensor(BinaryOps.elementwise(tensorA.core, tensorB.core, forwardOperation, dA, dB));
  }

  // UNARY OPERATIONS ---------------------------------

  /**
   * Applies an isolated unary operation function across elements of this tensor structure.
   *
   * @param forwardOperation the target operator function transformation sequence.
   * @param derivative the derivative function to find the local derivative, which is then multiplied by incoming gradients for the final result.
   * @return a new transformed tensor layout sequence.
   */
  public Tensor elementwise(UnaryOp forwardOperation, UnaryOp derivative) {
    return elementwise(tensor, forwardOperation, derivative);
  }

  /**
   * Applies an isolated unary operation function across elements of a specified
   * tensor.
   *
   * @param tensor the tensor to modify.
   * @param forwardOperation the target operator function transformation sequence.
   * @param derivative the derivative function to find the local derivative, which is then multiplied by incoming gradients for the final result.
   * @return a new transformed tensor layout sequence.
   */
  public static Tensor elementwise(Tensor tensor, UnaryOp forwardOperation, UnaryOp derivative) {
    if (forwardOperation == null) Message.crash("internal", "forward operation is null", "tensor elementwise", "forward operation cannot not be null");
    if (derivative == null) Message.crash("internal", "derivative operation is null", "tensor elementwise", "derivative operation cannot not be null");
    
    return new Tensor(UnaryOps.elementwise(tensor.core, forwardOperation, derivative));
  }

  /**
   * Applies a sliced unary array mapping operation function sices of this tensor structure along specified axes.
   * 
   * @param operation the bulk array mapping.
   * @param derivative the derivative function to find the local derivative, which is then multiplied by incoming gradients for the final result.
   * @param axes axes to apply the mapping along.
   * @return a new transformed tensor sequence.
   */
  public Tensor slicedMap(Mapping operation, Mapping derivative, int... axes) {
    return new Tensor(UnaryOps.slicedMap(tensor.core, operation, derivative, axes));
  }

  /**
   * Applies a sliced unary array mapping operation function sices of a tensor structure along specified axes.
   * 
   * @param tensor the tensor to modify.
   * @param operation the bulk array mapping.
   * @param derivative the derivative function to find the local derivative, which is then multiplied by incoming gradients for the final result.
   * @param axes axes to apply the mapping along.
   * @return a new transformed tensor sequence.
   */
  public static Tensor slicedMap(Tensor tensor, Mapping operation, Mapping derivative, int... axes) {
    if (operation == null) Message.crash("internal", "operation is null", "tensor slicedMap", "operation cannot not be null");
    if (derivative == null) Message.crash("internal", "derivative operation is null", "tensor slicedMap", "derivative operation cannot not be null");
    if (axes == null) Message.crash("internal", "axes is null", "tensor slicedMap", "axes cannot not be null");
    if (axes.length == 0) Message.crash("shaping", "axes is empty", "tensor slicedMap", "axes cannot be empty");
    Axes.verifyAxes("sliced mapping", tensor.getRank(), axes);

    return new Tensor(UnaryOps.slicedMap(tensor.core, operation, derivative, axes));
  }

  // SUBTENSOR OPERATIONS ---------------------------------

  /**
   * Expands this tensor along specified axes up to specified sizes using a given expansion operation. These new dimansions are computed by processing this tensor's elementwise and
   * naively filling (row-major order) the new axes with the result of the expansion operation.
   * 
   * Derivatives must be supplied to find the local derivative of the expansion function; Tensor will not check for gradient correctness.
   * 
   * @param operation the Expansion operation (double to double[]).
   * @param derivative a reduction function to find the local derivative.
   * @param newSizes the new sizes of the new dimension.
   * @param newAxes the location of the new axes in the expanded shape.
   * @return a new expanded tensor.
   */
  public Tensor expand(Expansion operation, Reduction derivative, int[] newSizes, int... newAxes) { 
    return expand(tensor, operation, derivative, newSizes, newAxes); 
  }

  /**
   * Expands a tensor along specified axes up to specified sizes using a given expansion operation. These new dimansions are computed by processing the old tensor elementwise and
   * naively filling (row-major order) the new axes with the result of the expansion operation.
   * 
   * Derivatives must be supplied to find the local derivative of the expansion function; Tensor will not check for gradient correctness.
   * 
   * @param tensor the tensor to expand.
   * @param operation the Expansion operation (double to double[]).
   * @param derivative a reduction function to find the local derivative.
   * @param newSizes the new sizes of the new dimension.
   * @param newAxes the location of the new axes in the expanded shape.
   * @return a new expanded tensor.
   */
  public static Tensor expand(Tensor tensor, Expansion operation, Reduction derivative, int[] newSizes, int... newAxes) { 
    if (operation == null) Message.crash("internal", "operation is null", "tensor expand", "operation cannot not be null");
    if (derivative == null) Message.crash("internal", "derivative operation is null", "tensor expand", "derivative operation cannot not be null");
    if (newSizes == null) Message.crash("internal", "newSizes is null", "tensor expand", "new sizes cannot not be null");
    if (newAxes == null) Message.crash("internal", "newAxes is null", "tensor expand", "new axes cannot not be null");
    if (newAxes.length == 0) Message.crash("shaping", "newAxes is empty", "tensor expand", "new axes cannot be empty");
    if (newSizes.length == 0) Message.crash("shaping", "newSizes is empty", "tensor expand", "new sizes cannot be empty");

    return new Tensor(SubtensorOps.expand(tensor.core, operation, derivative, newSizes, newAxes)); 
  }

  /**
   * Reduces this tensor along specified axes using a given reduction operation. Derivatives must be supplied to find the local derivative of the reduction function; Tensor
   * will not check for gradient correctness.
   *
   * @param operation the reduction function to apply. Has the form double[] -> double
   * @param derivative the derivative of the reduction function, applied to the orginal tensor. This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product). Has the form double[] -> double[]
   * @param axes the dimensions to reduce across.
   * @return a new reduced tensor.
   */
  public Tensor reduce(Reduction operation, Mapping derivative, int... axes) {
    return reduce(tensor, operation, derivative, axes);
  }

  /**
   * Reduces this tensor along specified axes using a given reduction operation. Derivatives must be supplied to find the local derivative of the reduction function; Tensor
   * will not check for gradient correctness.
   *
   * @param tensor the tensor to apply the operation to.
   * @param operation the reduction function to apply. Has the form double[] -> double
   * @param derivative the derivative of the reduction function, applied to the orginal tensor. This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product). Has the form double[] -> double[]
   * @param axes the dimensions to reduce across.
   * @return a new reduced tensor.
   */
  public static Tensor reduce(Tensor tensor, Reduction operation, Mapping derivative, int... axes) {
    if (operation == null) Message.crash("internal", "operation is null", "tensor reduce", "operation cannot not be null");
    if (derivative == null) Message.crash("internal", "derivative operation is null", "tensor reduce", "derivative operation cannot not be null");
    if (axes == null) Message.crash("internal", "axes is null", "tensor reduce", "axes cannot not be null");
    if (axes.length == 0) Message.crash("shaping", "axes is empty", "tensor reduce", "axes cannot be empty");
    Axes.verifyAxes("reduce", tensor.getRank(), axes);

    return new Tensor(SubtensorOps.reduce(tensor.core, operation, derivative, axes));
  }
}
