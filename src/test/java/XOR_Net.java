import java.util.ArrayList;
import java.util.List;

import com.aufy.jnet.jtensor.Tensor;
import com.aufy.jnet.net.Module;
import com.aufy.jnet.net.functions.LeakyReLU;
import com.aufy.jnet.net.functions.Softmax;
import com.aufy.jnet.net.layers.Dense;
import com.aufy.jnet.net.losses.CategoricalCrossEntropy;
import com.aufy.jnet.net.optimizers.SGD;

class Model extends Module {
  public Module layer1;
  public Module layer2;
  public Module layer3;
  public Module layer4;

  public Model() {
    this.layer1 = new Dense(2, 3);
    this.layer2 = new LeakyReLU(0.1);
    this.layer3 = new Dense(3, 2);
    this.layer4 = new Softmax();
  }

  @Override
  public Tensor forward(Tensor input) {
    Tensor x = input;

    x = layer1.forward(x);
    x = layer2.forward(x);
    x = layer3.forward(x);
    x = layer4.forward(x);

    return x;
  }

  @Override
  public List<Tensor> parameters() {
    List<Tensor> params = new ArrayList<>();

    params.addAll(layer1.parameters());
    params.addAll(layer2.parameters());
    params.addAll(layer3.parameters());
    params.addAll(layer4.parameters());

    return params;
  }
}

public class XOR_Net {
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

    Model model = new Model();

    CategoricalCrossEntropy loss = new CategoricalCrossEntropy(false);
    SGD optimizer = new SGD(model.parameters(), 0.1, 0.0);

    long startTime = System.nanoTime();
    for (int epoch = 0; epoch < 2000; epoch++) {
      Tensor prediction = model.forward(x);
      Tensor l = loss.compute(prediction, y);

      optimizer.zeroGrad();
      l.backward();
      optimizer.step();

      if (epoch % 100 == 0) {
        System.out.println("Epoch " + epoch + " | loss: " + l.dump(0));
      }
    }

    System.out.println(model.forward(x));

    double durationMs = (System.nanoTime() - startTime) / 1_000_000_000.0;
    System.out.println("Execution time (Seconds): " + durationMs);
  }
}
