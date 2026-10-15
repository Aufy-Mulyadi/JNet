package com.aufy.jnet.net;

import java.util.Collections;
import java.util.List;

import com.aufy.jnet.jtensor.Tensor;

/**
 * Base class for all modules (Layers & Activation functions).
 */
public abstract class Module {
  
  /**
   * Forward pass.
   * 
   * @param input the input tensor.
   * @return the output tensor.
   */
  public abstract Tensor forward(Tensor input);

  /**
   * Returns all trainable parameters.
   * 
   * @return trainable parameters.
   */
  public List<Tensor> parameters() {
    return Collections.emptyList();
  }

  public void train() {}

  public void eval() {}
}
