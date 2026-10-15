package com.aufy.jnet.net.functional;

import java.util.Arrays;

import com.aufy.jnet.core.backend.arrayops.Uniform;
import com.aufy.jnet.jtensor.Tensor;

public class Extra {
  public static Tensor softmax(Tensor tensor, int... axes) {
    return tensor.ops.slicedMap(
      Uniform::softmax, 
      (double[] derivative) -> {
        Arrays.fill(derivative, 1.0);
        return derivative;
      }, 1
    );
  }
  
  public static Tensor logSoftmax(Tensor tensor, int... axes) {
    return tensor.sub(logSumExp(tensor, axes));
  }

  public static Tensor logSumExp(Tensor tensor, int... axes) {
    return tensor.reduce.logSumExp(axes);
  }
}
