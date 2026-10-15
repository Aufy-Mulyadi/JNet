package com.aufy.jnet.jtensor;

import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

public class Custom {
  private final Tensor tensor;

  public Custom(Tensor tensor) {
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


  // UNARY OPERATIONS ---------------------------------

  /**
   * Applies a unary operation to the specified tensor. Autograd ignores everything inside forwardOperation and uses derivative to propagate gradients.
   * 
   * Tensor will not double-check gradient correctness.
   * 
   * @param forwardOperation forward operation applied to tensor (symbolically) as operation.apply(tensor).
   * @param derivative backward operation applied to the incoming gradient tensor as derivative.apply(grad). This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product).
   * @return this Tensor after the operation has been applied.
   */
  public Tensor apply(Function<Tensor, Tensor> forwardOperation, Function<Tensor, Tensor> derivative) {
    return apply(tensor, forwardOperation, derivative); 
  }

  /**
   * Applies a unary operation to the specified tensor. Autograd ignores everything inside forwardOperation and uses derivative to propagate gradients.
   * 
   * Tensor will not double-check gradient correctness.
   * 
   * @param tensor the tensor to apply the operation to.
   * @param forwardOperation forward operation applied to tensor (symbolically) as operation.apply(tensor).
   * @param derivative backward operation applied to the incoming gradient tensor as derivative.apply(grad). This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product).
   * @return a Tensor after the operation has been applied.
   */
  public static Tensor apply(Tensor tensor, Function<Tensor, Tensor> forwardOperation, Function<Tensor, Tensor> derivative) {
    Tensor out = forwardOperation.apply(tensor);

    if (tensor.requiresGrad) out.requiresGrad();

    if (out.requiresGrad) {
      out.core.parents = List.of(tensor.core);

      out.core.derivative = (grad) -> {
        Tensor derivativeCore = derivative.apply(new Tensor(grad));

        tensor.core.accumulate(Tensor.hadamard(new Tensor(grad), derivativeCore).core.core);
      };
    }

    return out; 
  }

  // BINARY OPERATIONS ---------------------------------

  /**
   * Applies a binary operation to the specified tensors. Autograd ignores everything inside forwardOperation and uses dA and dB to propagate partial gradients with respect to each tensor respectively.
   * 
   * Tensor will not double-check gradient correctness.
   * 
   * @param tensorB the second tensor to apply the operation to.
   * @param forwardOperation forward operation applied to tensor (symbolically) as operation.apply(tensorA, tensorB).
   * @param dA partial derivative function of the incoming gradient tensor with respect to the first tensor applied as dA.apply(tensorB, grad). This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product).
   * @param dB partial derivative function of the incoming gradient tensor with respect to the second tensor applied as dB.apply(tensorA, grad). This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product).
   * @return this Tensor after the operation has been applied.
   */
  public Tensor apply(Tensor tensorB, BiFunction<Tensor, Tensor, Tensor> forwardOperation, BiFunction<Tensor, Tensor, Tensor> dA, BiFunction<Tensor, Tensor, Tensor> dB) {
    return apply(tensor, tensorB, forwardOperation, dA, dB); 
  }

  /**
   * Applies a binary operation to the specified tensors. Autograd ignores everything inside forwardOperation and uses dA and dB to propagate partial gradients with respect to each tensor respectively.
   * 
   * Tensor will not double-check gradient correctness.
   * 
   * @param tensorA the first tensor to apply the operation to.
   * @param tensorB the second tensor to apply the operation to.
   * @param forwardOperation forward operation applied to tensor (symbolically) as operation.apply(tensorA, tensorB).
   * @param dA partial derivative function of the incoming gradient tensor with respect to the first tensor applied as dA.apply(tensorB, grad). This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product).
   * @param dB partial derivative function of the incoming gradient tensor with respect to the second tensor applied as dB.apply(tensorA, grad). This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product).
   * @return a Tensor after the operation has been applied.
   */
  public static Tensor apply(Tensor tensorA, Tensor tensorB, BiFunction<Tensor, Tensor, Tensor> forwardOperation, BiFunction<Tensor, Tensor, Tensor> dA, BiFunction<Tensor, Tensor, Tensor> dB) {
    Tensor out = forwardOperation.apply(tensorA, tensorB);
    
    if (tensorA.requiresGrad || tensorB.requiresGrad) out.requiresGrad();
    
    if (out.requiresGrad) {
      out.core.parents = List.of(tensorA.core, tensorB.core);
      
      out.core.derivative = (grad) -> {
        if (tensorA.requiresGrad) {
          Tensor dACore = dA.apply(tensorA, tensorB);
          tensorA.core.accumulate(Tensor.hadamard(new Tensor(grad), dACore).core.core);
        }
        
        if (tensorB.requiresGrad) {
          Tensor dBCore = dB.apply(tensorA, tensorB);
          tensorA.core.accumulate(Tensor.hadamard(new Tensor(grad), dBCore).core.core);
        }
      };
    }
    
    return out;
  }

}
