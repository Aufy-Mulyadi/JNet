

import java.util.List;

import com.aufy.jnet.jtensor.Tensor;
import com.aufy.jnet.net.functional.Regularizer;
import com.aufy.jnet.net.optimizers.Optimizer;

public class AdaGrad extends Optimizer {
  private final Regularizer regularizer;

  private final Tensor[] accumulator;

  public AdaGrad(List<Tensor> params) {
    this(params, null);
  }

  public AdaGrad(List<Tensor> params, Regularizer regularizer) {
    super(params, 0);
    this.regularizer = regularizer;

    this.accumulator = new Tensor[params.size()];

    for (int i = 0; i < params.size(); i++) accumulator[i] = Tensor.init(params.get(i)).zeros();
  }

  @Override
  public void step() {
    for (int i = 0; i < parameters.size(); i++) {
      Tensor p = parameters.get(i);

      if (regularizer != null) {
        regularizer.updateGrad(p.grad());
      }

      accumulator[i].inplace.add(p.grad().inplace.pow(2));

      p.inplace.sub(
        p.grad()
        .inplace.mul(learningRate)
        .inplace.div(
          accumulator[i].inplace.root().inplace.add(1e-8)
        )
      );
    }
  }
}