package com.aufy.jnet.core.tensor.raw.main;

import java.util.Arrays;

import com.aufy.jnet.core.backend.interfaces.atomic.BinaryOp;
import com.aufy.jnet.core.tensor.core.backend.compute.Engine;
import com.aufy.jnet.core.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;

public class RawBinaryOps {

  public static RawTensor elementwise(RawTensor a, RawTensor b, BinaryOp operation) {
    if (!Arrays.equals(a.shape, b.shape)) {
      int[] broadcastShapeTarget = Shaping.broadcastedShape(a.shape, b.shape);

      a = RawShapeOps.broadcast(a, broadcastShapeTarget);
      b = RawShapeOps.broadcast(b, broadcastShapeTarget);
    }

    double[] resultData = BinaryOp.apply(a.dump(), b.dump(), operation);
    return new RawTensor(resultData, a.shape);
  }

  public static RawTensor contract(RawTensor a, RawTensor b, int[] axesA, int[] axesB) {
    int[] resShape = Shaping.calculateResShape(a.shape, b.shape, axesA, axesB);
    
    double[] resultData = Engine.contract(a.dump(), a.strides, b.dump(), b.strides, a.shape, axesA, b.shape, axesB, resShape);

    return new RawTensor(resultData, resShape);
  }
  
}
