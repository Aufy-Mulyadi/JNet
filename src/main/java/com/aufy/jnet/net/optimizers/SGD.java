package com.aufy.jnet.net.optimizers;

import java.util.List;

import com.aufy.jnet.jtensor.Tensor;
import com.aufy.jnet.net.functional.Regularizer;

public class SGD extends Optimizer {
  private final double momentum;
  private final Regularizer regularizer;

  private final Tensor[] velocity;

  public SGD(List<Tensor> params, double learningRate, double momentum) {
    this(params, learningRate, momentum, null);
  }

  public SGD(List<Tensor> params, double learningRate, double momentum, Regularizer regularizer) {
    super(params, learningRate);
    this.momentum = momentum;
    this.regularizer = regularizer;
    this.velocity = new Tensor[params.size()];

    for (int i = 0; i < params.size(); i++) velocity[i] = Tensor.init(params.get(i)).zeros();
  }

  @Override
  public void step() {
    for (int i = 0; i < parameters.size(); i++) {
      Tensor p = parameters.get(i);
      
      if (this.regularizer != null) {
        regularizer.updateGrad(p.grad());
      }

      if (this.momentum != 0) { // momentum
        velocity[i].inplace.mul(this.momentum); // b * g_t-1
        p.grad().inplace.mul(1 - this.momentum); // (1 - b) * g
        velocity[i].inplace.add(p.grad()); // momentum
        
        p.inplace.sub(velocity[i].inplace.mul(this.learningRate));
      } else { // sgd
        p.inplace.sub(p.grad().inplace.mul(this.learningRate));
      }
    }
  }
}
