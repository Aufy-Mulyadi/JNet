package com.aufy.jnet.net.functions;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.jtensor.Tensor;
import com.aufy.jnet.net.Module;
import com.aufy.jnet.net.functional.Extra;

public class Logsumexp extends Module {
  private final int[] axes;

  public Logsumexp(int... axes) {
    this.axes = axes;
  }

  @Override
  public Tensor forward(Tensor x) {
    Tensor adjusted = x.add(-Reductions.max(x.dump()));

    return adjusted.custom.apply(
      tensor -> tensor.exp().reduce.sum(this.axes).ln(), 
      tensor -> Extra.softmax(tensor, this.axes)
    );
  }
}
