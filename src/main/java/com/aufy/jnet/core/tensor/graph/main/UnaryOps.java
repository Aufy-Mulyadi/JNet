package com.aufy.jnet.core.tensor.graph.main;

import java.util.List;

import com.aufy.jnet.core.backend.interfaces.atomic.BinaryOp;
import com.aufy.jnet.core.backend.interfaces.atomic.UnaryOp;
import com.aufy.jnet.core.backend.interfaces.composite.Mapping;
import com.aufy.jnet.core.tensor.core.implementations.CoreTensor;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;
import com.aufy.jnet.core.tensor.raw.main.RawUnaryOps;

public class UnaryOps {

  public static CoreTensor slicedMap(CoreTensor tensor, Mapping operation, Mapping derivative, int[] axes) {
    CoreTensor out = new CoreTensor(RawUnaryOps.slicedMap(tensor.core, operation, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor derivativeCore = RawUnaryOps.slicedMap(tensor.core, derivative, axes); // dy/dx
        // RawTensor gradInputCore = RawBinaryOps.elementwise(grad, derivativeCore, (a, b) -> a * b); // grad * dy/dx = grad/dx
        
        tensor.accumulate(BinaryOp.apply(grad.data, derivativeCore.data, (a, b) -> a * b));
      };
    }

    return out;
  }

  public static CoreTensor elementwise(CoreTensor tensor, UnaryOp operation, UnaryOp derivative) {
    CoreTensor out = new CoreTensor(RawUnaryOps.elementwise(tensor.core, operation));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor derivativeCore = RawUnaryOps.elementwise(tensor.core, derivative); // dy/dx
        
        tensor.accumulate(BinaryOp.apply(grad.data, derivativeCore.data, (a, b) -> a * b));
      };
    }

    return out;
  }
}
