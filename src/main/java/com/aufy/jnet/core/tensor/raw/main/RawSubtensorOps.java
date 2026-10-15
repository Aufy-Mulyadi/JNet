package com.aufy.jnet.core.tensor.raw.main;

import com.aufy.jnet.core.backend.arrayops.Tools;
import com.aufy.jnet.core.backend.interfaces.composite.Expansion;
import com.aufy.jnet.core.backend.interfaces.composite.Reduction;
import com.aufy.jnet.core.tensor.core.backend.compute.Engine;
import com.aufy.jnet.core.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;

public class RawSubtensorOps {

  public static RawTensor reduce(RawTensor tensor, Reduction operation, int[] axes) {
    return new RawTensor(
      Engine.reduce(tensor.dump(), tensor.shape, tensor.strides, axes, operation::apply), 
      Shaping.reduceAxes(tensor.shape, axes)
    );
  }

  public static RawTensor expand(RawTensor tensor, Expansion operation, int[] newSizes, int[] newAxes) {
    RawTensor unsqueezed = RawShapeOps.unsqueeze(tensor, newAxes);
    RawTensor broadcasted = RawShapeOps.broadcast(unsqueezed, Tools.scatter(tensor.shape, newAxes, newSizes));

    return RawUnaryOps.slicedMap(broadcasted, data -> operation._apply(data[0]), newAxes);
  }

}
