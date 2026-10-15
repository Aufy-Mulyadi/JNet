import com.aufy.jnet.jtensor.Tensor;

public class gradientTensor {
  public static void main(String[] args) {
    Tensor x = new Tensor(new double[] {
      0f, 0f,
      0f, 1f,
      1f, 0f,
      1f, 1f
    }, 4, 2);

    // Tensor weight = new Tensor(new Gaussian(0, 1), 3, 2).requiresGrad();
    Tensor weight = Tensor.init(3, 2).normal(0, 1).requiresGrad();
    Tensor bias = Tensor.init(1, 3).zeros().requiresGrad();

    Tensor C = x.matmul(weight.transpose()).add(bias);
    Tensor D = C.ops.elementwise(
      a -> Math.max(a, a * 0.1f), 
      a -> (a >= 0)? 1.0f : 0.1f
    );
    Tensor E = D.ops.elementwise(
      a -> Math.max(a, a * 0.1f), 
      a -> (a >= 0)? 1.0f : 0.1f
    );

    E.backward();
    System.out.println(weight.grad());
  }
}
