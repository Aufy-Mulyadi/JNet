package com.aufy.jnet.net.optimizers;

import java.util.List;

import com.aufy.jnet.jtensor.Tensor;

public abstract class Optimizer {
  public final List<Tensor> parameters;
  protected double learningRate;

  public Optimizer(List<Tensor> parameters, double learningRate) {
    this.parameters = parameters;
    this.learningRate = learningRate;
  }

  public abstract void step();

  public void zeroGrad() {
    for (Tensor p : parameters) p.zeroGrad();
  }

}
