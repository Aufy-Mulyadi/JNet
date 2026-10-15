package com.aufy.jnet.net.functions;

import com.aufy.jnet.jtensor.Tensor;
import com.aufy.jnet.net.Module;

public class ReLU extends Module {

  @Override
  public Tensor forward(Tensor x) {
    return x.ops.elementwise(
      a -> Math.max(a, 0.0), 
      a -> (a > 0)? 1.0 : 0.0
    );
  }
}
