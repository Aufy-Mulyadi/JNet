package com.aufy.jnet.net.losses;

import com.aufy.jnet.jtensor.Custom;
import com.aufy.jnet.jtensor.Tensor;
import com.aufy.jnet.net.functional.Extra;

public class CategoricalCrossEntropy extends Loss{
  private final boolean logits;

  public CategoricalCrossEntropy(boolean logits) {
    this.logits = logits;
  }

  @Override
  public Tensor compute(Tensor prediction, Tensor target) {
    return Custom.apply(prediction, target,
      (p, t) -> {
        Tensor logsoftmax;

        if (logits) {
          logsoftmax = Extra.logSoftmax(p, 1);
        } else {
          logsoftmax = p.ln();
        }

        Tensor loss = logsoftmax.mul(-1.0).hadamard(t);

        return loss.stats(0,1).mean();
      },

      (p, t) -> {
        Tensor softmax;

        if (logits) {
          softmax = Extra.softmax(p, 1);
        } else {
          softmax = p;
        }

        return softmax.sub(t);
      },
      (p, t) -> p.init().ones() // target wont need gradients
    );
  }
}
