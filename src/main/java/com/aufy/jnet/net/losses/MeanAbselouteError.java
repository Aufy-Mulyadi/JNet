package com.aufy.jnet.net.losses;

import com.aufy.jnet.jtensor.Tensor;

public class MeanAbselouteError extends Loss{
  @Override
  public Tensor compute(Tensor prediction, Tensor target) {
    int size = prediction.getSize();
    return prediction.sub(target).util.abs().reduce.sum(0).mul(1.0/size);
  }
}
