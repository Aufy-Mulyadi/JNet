package com.aufy.jnet.core.tensor.raw.main;

import com.aufy.jnet.core.backend.interfaces.atomic.UnaryOp;
import com.aufy.jnet.core.backend.interfaces.composite.Mapping;
import com.aufy.jnet.core.tensor.core.backend.compute.Engine;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;

public class RawUnaryOps {
  
  public static RawTensor elementwise(RawTensor tensor, UnaryOp operation) {
    return new RawTensor(UnaryOp.apply(tensor.dump(), operation), tensor.shape);
  }

  public static RawTensor slicedMap(RawTensor tensor, Mapping operation, int[] axes) {
    return new RawTensor(
      Engine.slicedMap(tensor.dump(), tensor.shape, tensor.strides, axes, operation::apply), 
      tensor.shape
    );
  }
}
