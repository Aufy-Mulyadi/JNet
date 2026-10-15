package com.aufy.jnet.net.functional;

import com.aufy.jnet.core.net.functional.Encoders;
import com.aufy.jnet.jtensor.Tensor;

public class Encoder {
  
  /**
   * One-hot encodes a tensor, only considering a specific slice for evaluation. Expanded dimension is the last dimension.
   * 
   * @param input input tensor.
   * @param classes positions of classes to encode.
   * @param axes axes to consider.
   * @return encoded tensor.
   */
  public static Tensor oneHot(Tensor input, double[] classes) {
    return input.ops.expand(
      value -> Encoders.onehot(value, classes), 
      value -> 0, 
      new int[]{classes.length},
      input.getRank()
    );
  }

}
