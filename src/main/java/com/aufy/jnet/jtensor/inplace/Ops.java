package com.aufy.jnet.jtensor.inplace;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.exceptions.Message;
import com.aufy.jnet.core.backend.interfaces.atomic.BinaryOp;
import com.aufy.jnet.core.backend.interfaces.atomic.UnaryOp;
import com.aufy.jnet.core.backend.interfaces.composite.Mapping;
import com.aufy.jnet.core.tensor.core.backend.compute.Engine;
import com.aufy.jnet.jtensor.Internal;
import com.aufy.jnet.jtensor.Tensor;

public class Ops {
  private final Tensor tensor;

  public Ops(Tensor tensor) {
    this.tensor = tensor;
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
   * Performs an elementwise binary operation on this tensor and another tensor.
   *
   * @param tensorB the second tensor involved in the operation.
   * @param forwardOperation the binary function to apply to elements.
   * @param dA the partial derivative function with respect to the first tensor.
   * @param dB the partial derivative function with respect to the second tensor.
   * @return a new tensor containing the results of the elementwise operation.
   * @return a reference to this tensor.
   */
  public Tensor elementwise(Tensor tensorB, BinaryOp forwardOperation) {
    elementwise(tensor, tensorB, forwardOperation);
    return tensor;
  }

  /**
   * Performs an elementwise binary operation between two specified tensors.
   *
   * @param tensorA the first tensor.
   * @param tensorB the second tensor.
   * @param forwardOperation the binary function to apply to elements.
   * @param dA the partial derivative function with respect to tensorA.
   * @param dB the partial derivative function with respect to tensorB.
   * @return a new tensor containing the results of the elementwise operation.double
   * @return a reference to the tensor.
   */
  public static Tensor elementwise(Tensor tensorA, Tensor tensorB, BinaryOp forwardOperation) {
    if (tensorA.getRank() != tensorB.getRank()) Message.crash("internal", "tensors have differing ranks", "in-place tensor elementwise", "both tensors must have the same rank");
    if (Reductions.prod(tensorA.getShape()) != Reductions.prod(tensorB.getShape())) Message.crash("internal", "tensors have different volumes", "in-place tensor elementwise", "in-place elementwise operations does not support broadcasting; both tensors must have the same shape");
    if (forwardOperation == null) Message.crash("internal", "forward operation is null", "in-place tensor elementwise", "forward operation cannot not be null");
    
    Internal.setData(tensorA, BinaryOp.apply_(Internal.getData(tensorA), Internal.getData(tensorB), forwardOperation));

    return tensorA;
  }

  // UNARY OPERATIONS ---------------------------------

  /**
   * Applies an isolated unary operation function across elements of this tensor structure.
   *
   * @param forwardOperation the target operator function transformation sequence.
   * @param derivative the derivative function to find the local derivative, which is then multiplied by incoming gradients for the final result.
   * @return a new transformed tensor layout sequence.
   * @return a reference to this tensor.
   */
  public Tensor elementwise(UnaryOp forwardOperation) {
    elementwise(tensor, forwardOperation);
    return tensor;
  }

  /**
   * Applies an isolated unary operation function across elements of a specified
   * tensor.
   *
   * @param tensor the tensor to modify.
   * @param forwardOperation the target operator function transformation sequence.
   * @param derivative the derivative function to find the local derivative, which is then multiplied by incoming gradients for the final result.
   * @return a new transformed tensor layout sequence.
   * @return a reference to the tensor.
   */
  public static Tensor elementwise(Tensor tensor, UnaryOp forwardOperation) {
    if (forwardOperation == null) Message.crash("internal", "forward operation is null", "in-place tensor elementwise", "forward operation cannot not be null");
    
    Internal.setData(tensor, UnaryOp.apply(Internal.getData(tensor), forwardOperation));
    return tensor;
  }

  /**
   * Applies a sliced unary array mapping operation function sices of this tensor structure along specified axes.
   * 
   * @param operation the bulk array mapping.
   * @param derivative the derivative function to find the local derivative, which is then multiplied by incoming gradients for the final result.
   * @param axes axes to apply the mapping along.
   * @return a new transformed tensor sequence.
   * @return a reference to this tensor.
   */
  public Tensor slicedMap(Mapping operation, int... axes) {
    slicedMap(tensor, operation, axes);
    return tensor;
  }

  /**
   * Applies a sliced unary array mapping operation function sices of this tensor structure along specified axes.
   * 
   * @param tensor the tensor to modify.
   * @param operation the bulk array mapping.
   * @param derivative the derivative function to find the local derivative, which is then multiplied by incoming gradients for the final result.
   * @param axes axes to apply the mapping along.
   * @return a new transformed tensor sequence.
   * @return a reference to the tensor.
   */
  public static Tensor slicedMap(Tensor tensor, Mapping operation, int... axes) {
    Internal internal = new Internal(tensor);

    double[] mappedData = Engine.slicedMap(
      internal.data, 
      internal.shape, 
      internal.strides, 
      axes,
      operation::_apply
    );

    internal.setData(mappedData);
    return tensor;
  }
}
