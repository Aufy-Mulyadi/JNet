package com.aufy.jnet.net.optimizers;

import java.util.List;

import com.aufy.jnet.jtensor.Tensor;
import com.aufy.jnet.net.functional.Regularizer;

public class ADAM extends Optimizer {
  private final Regularizer regularizer;
  private final double beta1;
  private final double beta2;
  private int t=0;

  private final Tensor[] m;
  private final Tensor[] v;

  public ADAM(List<Tensor> params, double learningRate, double beta1, double beta2) {
    this(params, learningRate, beta1, beta2, null);
  }

  public ADAM(List<Tensor> params, double learningRate, double beta1, double beta2, Regularizer regularizer) {
    super(params, learningRate);
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.regularizer = regularizer;
    this.m = new Tensor[params.size()];
    this.v = new Tensor[params.size()];

    for (int i = 0; i < params.size(); i++) {
      m[i] = Tensor.init(params.get(i)).zeros();
      v[i] = Tensor.init(params.get(i)).zeros();
    }
  }

  @Override
  public void step() {
    t++;
    for (int i = 0; i < parameters.size(); i++) {
      Tensor p = parameters.get(i);
      
      if (this.regularizer != null) {
        regularizer.updateGrad(p.grad());
      }

      m[i].inplace.mul(this.beta1).inplace.add(p.grad().inplace.mul(1 - this.beta1)); // betas
      v[i].inplace.mul(this.beta2).inplace.add(p.grad().inplace.pow(2).inplace.mul(1 - this.beta2));

      Tensor mtHat = m[i].inplace.div(1 - Math.pow(this.beta1, t)); // bias correction
      Tensor vtHat = v[i].inplace.div(1 - Math.pow(this.beta2, t));

      p.inplace.sub(
        mtHat.inplace.mul(this.learningRate)
        .inplace.div(vtHat.inplace.root(2).inplace.add(1e-8) )
      );
    }
  }
}
