package com.aufy.jnet.net.metrics;

import com.aufy.jnet.jtensor.Tensor;

public final class Accuracy extends Metric {

  @Override
  public double compute(Tensor prediction, Tensor target) {
    Tensor predicted = prediction.reduce.argmax(1).squeeze(1);
    double correct = predicted.logic.equal(target).reduce.sum(0).dump()[0];
    return correct / target.getShape(0);
  }
}