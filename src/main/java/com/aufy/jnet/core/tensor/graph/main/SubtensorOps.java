package com.aufy.jnet.core.tensor.graph.main;

import java.util.List;

import com.aufy.jnet.core.backend.interfaces.composite.Expansion;
import com.aufy.jnet.core.backend.interfaces.composite.Mapping;
import com.aufy.jnet.core.backend.interfaces.composite.Reduction;
import com.aufy.jnet.core.tensor.core.implementations.CoreTensor;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;
import com.aufy.jnet.core.tensor.raw.main.RawBinaryOps;
import com.aufy.jnet.core.tensor.raw.main.RawShapeOps;
import com.aufy.jnet.core.tensor.raw.main.RawSubtensorOps;
import com.aufy.jnet.core.tensor.raw.main.RawUnaryOps;

public class SubtensorOps {
  
  public static CoreTensor expand(CoreTensor tensor, Expansion operation, Reduction derivative, int[] newAxes, int[] newSizes) {
    CoreTensor out = new CoreTensor(RawSubtensorOps.expand(tensor.core, operation, newAxes, newSizes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        // remove the extra dimension
        RawTensor localDerivative = RawShapeOps.squeeze(RawSubtensorOps.reduce(tensor.core, derivative, newAxes), newAxes);

        tensor.accumulate(RawBinaryOps.elementwise(localDerivative, grad, (a, b) -> a * b));
      };
    }

    return out;
  }

  public static CoreTensor reduce(CoreTensor tensor, Reduction operation, Mapping derivative, int[] axes) {
    CoreTensor out = new CoreTensor(RawSubtensorOps.reduce(tensor.core, operation, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor expandedGrads = RawShapeOps.broadcast(grad, tensor.shape);

        RawTensor localDerivative = RawUnaryOps.slicedMap(tensor.core, derivative, axes);
        tensor.accumulate(RawBinaryOps.elementwise(localDerivative, expandedGrads, (a, b) -> a * b));
      };
    }

    return out;
  }

}
