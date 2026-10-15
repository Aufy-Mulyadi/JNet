package com.aufy.jnet.net.functional;

import java.util.List;

import com.aufy.jnet.jtensor.Tensor;

/**
 * ElasticNet regularization.
 * 
 * ElasticNet is a regularization method that combines L1 (Lasso) and L2 (Ridge) regularization. 
 * Contains methods to compute the regularization penalty loss and updates gradient tensors in-place during the backward pass.
 */
public class Regularizer {
  private final double alpha;
  private final double beta;
  private final List<Tensor> params;

  public Regularizer(List<Tensor> params, double alpha, double beta) {
    this.alpha = alpha;
    this.beta = beta;
    this.params = params;
  }

  /**
   * Computes the additional loss to be added to the base loss value.
   * 
   * @return the additional loss.
   */
  public double compute() {
    double l1Total = 0;
    double l2Total = 0;

    for (Tensor param : params) {
      l1Total += param.stats().norm(1).dump(0);
      l2Total += param.stats().norm(2).dump(0);
    }

    return alpha * (beta * l1Total + (1 - beta) * l2Total);
  }

  /**
   * Applies the regularizer to the specified gradient tensor and modifies it in-place.
   * 
   * @param grad the gradient tensor.
   */
  public void updateGrad(Tensor grad) {
    grad.inplace.add(
    grad.util_.sign().inplace.mul(beta).inplace.add(grad.inplace.mul(1 - beta)).inplace.mul(alpha)
    );
  }
}
