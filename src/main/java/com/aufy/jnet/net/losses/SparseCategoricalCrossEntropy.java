package com.aufy.jnet.net.losses;

import com.aufy.jnet.jtensor.Custom;
import com.aufy.jnet.jtensor.Tensor;
import com.aufy.jnet.net.functional.Encoder;
import com.aufy.jnet.net.functional.Extra;

public class SparseCategoricalCrossEntropy extends Loss{
  private final double[] classes;
  private final boolean logits;
  
  public SparseCategoricalCrossEntropy(boolean logits, double... classes) {
    this.classes = classes;
    this.logits = logits;
  }

  @Override
  public Tensor compute(Tensor prediction, Tensor target) {
    Tensor encoded = Encoder.oneHot(target, classes);
    
    return Custom.apply(prediction, encoded,
      (p, t) -> {
        Tensor logsoftmax;

        if (logits) {
          logsoftmax = Extra.logSoftmax(p, 1);
        } else {
          logsoftmax = p.ln();
        }

        Tensor loss = logsoftmax.hadamard(t).mul(-1).reduce.sum(1);

        return loss.stats().mean();
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
      (p, t) -> null // target wont need gradients
    );
  }
}
