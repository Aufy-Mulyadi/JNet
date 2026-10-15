package com.aufy.jnet.net.losses;

import com.aufy.jnet.core.backend.scalarops.Unary;
import com.aufy.jnet.jtensor.Custom;
import com.aufy.jnet.jtensor.Tensor;

public class BinaryCrossEntropy extends Loss{
  private final boolean logits;

  public BinaryCrossEntropy(boolean logits) {
    this.logits = logits;
  }

  @Override
  public Tensor compute(Tensor prediction, Tensor target) {
    return Custom.apply(prediction, target,
      (p, t) -> {
        Tensor probability = p;

        if (logits) {
          probability = p.ops.elementwise(Unary::sigmoid, null);
        }

        Tensor loss1 = probability.ln().hadamard(t);
        Tensor loss2 = p.init().ones().sub(probability).ln().hadamard(t.init().ones().sub(t));
        Tensor loss = loss1.add(loss2);

        return loss.stats(0,1).mean();
      },

      (p, t) -> {
        Tensor sigmoided = p;

        if (logits) {
          sigmoided = p.ops.elementwise(Unary::sigmoid, null);
        }

        return sigmoided.sub(t);
      },
      (p, t) -> null // target wont need gradients
    );
  }
}
