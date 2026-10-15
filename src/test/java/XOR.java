import com.aufy.jnet.api.Sequential;
import com.aufy.jnet.jtensor.Tensor;
import com.aufy.jnet.net.functions.LeakyReLU;
import com.aufy.jnet.net.functions.Softmax;
import com.aufy.jnet.net.layers.Dense;
import com.aufy.jnet.net.losses.CategoricalCrossEntropy;
import com.aufy.jnet.net.optimizers.SGD;

public class XOR {
  public static void main(String[] args) {
    Tensor x = new Tensor(new double[] {
      0, 0,
      0, 1,
      1, 0,
      1, 1
    }, 4, 2);

    Tensor y = new Tensor(new double[] {
      0, 1,
      1, 0,
      1, 0,
      0, 1
    }, 4, 2);

    Sequential model = new Sequential(
      new Dense(2, 3),
      new LeakyReLU(0.1),
      new Dense(3, 2),
      new Softmax()
    );

    CategoricalCrossEntropy loss = new CategoricalCrossEntropy(false);
    SGD optimizer = new SGD(model.parameters(),0.1,0.2);

    long startTime = System.nanoTime();
    for (int epoch = 0; epoch < 2000; epoch++) {
      Tensor prediction = model.forward(x);
      Tensor l = loss.compute(prediction, y);

      optimizer.zeroGrad();
      l.backward();
      optimizer.step();

      // System.out.println(epoch + " : " + l);
      // System.out.println(model.parameters().size());
      // System.out.println(optimizer.parameters.get(3).core);
    }

    System.out.println(model.forward(x));

    double durationMs = (System.nanoTime() - startTime) / 1_000_000_000.0;
    System.out.println("Execution time (Seconds): " + durationMs);
  }
}
