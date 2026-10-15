package com.aufy.jnet.net.layers;

import java.util.List;

import com.aufy.jnet.jtensor.Tensor;
import com.aufy.jnet.net.Module;

public class Dense extends Module {
  
  private final Tensor weight;
  private final Tensor bias;

  public Dense(int incomingDimensions, int neurons) {
    double stdev = Math.sqrt(2.0 / incomingDimensions);

    // temporary; add custom initializers soon
    weight = Tensor.init(neurons, incomingDimensions).normal(0, stdev).requiresGrad();
    bias = Tensor.init(1, neurons).zeros().requiresGrad();
  }

  @Override
  public Tensor forward(Tensor x) {
    return x.matmul(weight.transpose()).add(bias);
  }

  @Override
  public List<Tensor> parameters() {
    return List.of(weight, bias);
  }
}
