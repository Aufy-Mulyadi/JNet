package com.aufy.jnet.net.losses;

import com.aufy.jnet.jtensor.Custom;
import com.aufy.jnet.jtensor.Tensor;

public class MeanSquaredError extends Loss{
  @Override
  public Tensor compute(Tensor prediction, Tensor target) {
    int size = prediction.getSize();
    // return prediction.sub(target).pow(2).sum(0,1).mul(1.0/size);

    return Custom.apply(prediction, target,
      (p, t) -> {
        return p.sub(t).pow(2).reduce.sum(0,1).mul(1.0/size);
      },

      (p, t) -> p.sub(t).mul(2.0),
      (p, t) -> null // target wont need gradients
    );
  }
}
