package com.aufy.jnet.net.functions;

import java.util.Arrays;

import com.aufy.jnet.core.backend.arrayops.Uniform;
import com.aufy.jnet.jtensor.Tensor;
import com.aufy.jnet.net.Module;

public class Softmax extends Module {

  @Override
  public Tensor forward(Tensor x) {
    return x.ops.slicedMap(
      Uniform::softmax, 
      (double[] derivative) -> {
        Arrays.fill(derivative, 1.0);
        return derivative;
      }, 1
    );
    
  }
}
