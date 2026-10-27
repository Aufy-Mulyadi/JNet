import java.util.ArrayList;
import java.util.List;

import com.aufy.jnet.data.DataContainer;
import com.aufy.jnet.data.DataLoader;
import com.aufy.jnet.data.FileLoader;
import com.aufy.jnet.jtensor.Tensor;
import com.aufy.jnet.net.Module;
import com.aufy.jnet.net.functions.LeakyReLU;
import com.aufy.jnet.net.layers.Dense;
import com.aufy.jnet.net.losses.Loss;
import com.aufy.jnet.net.losses.SparseCategoricalCrossEntropy;
import com.aufy.jnet.net.metrics.Accuracy;
import com.aufy.jnet.net.metrics.Metric;
import com.aufy.jnet.net.optimizers.ADAM;

class model extends Module {
  public Module activation;
  public Module layer1;
  public Module layer2;
  public Module layer3;
  public Module layer4;
  
  public model() {
    this.activation = new LeakyReLU(0.1);
    this.layer1 = new Dense(784, 512);
    this.layer2 = new Dense(512, 64);
    this.layer3 = new Dense(64, 64);
    this.layer4 = new Dense(64, 10);
  }

  @Override
  public Tensor forward(Tensor input) {
    Tensor x = input;

    x = layer1.forward(x);
    x = activation.forward(x);
    x = layer2.forward(x);
    x = activation.forward(x);
    x = layer3.forward(x);
    x = activation.forward(x);
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

public class MNIST {

  public static void main(String[] args) {
    Module model = new model();

    /*
    60,000 training
    10,000 test
    Each row consists of 785 values: [label, pixel1, pixel2, ..., pixel784]
    - the first value is the label (a number from 0 to 9) 
    - the remaining 784 values are the pixel values (a number from 0 to 255).
    */
    DataContainer MNIST = FileLoader.loadCSV("src\\test\\resources\\mnist_test.csv", new int[]{0}, new int[]{}, new int[]{784});
    MNIST.features().inplace.div(255);
    DataLoader loader = new DataLoader(MNIST, 32, true);
    Loss loss = new SparseCategoricalCrossEntropy(true, 0,1,2,3,4,5,6,7,8,9);
    ADAM optimizer = new ADAM(model.parameters(),0.005, 0.9, 0.999);
    Metric metric = new Accuracy();
    
    // network stuff
    Tensor x; Tensor y; Tensor prediction; Tensor l; double accuracy;
    for (int epoch = 0; epoch < 10; epoch++) {
      x = MNIST.features();
      y = MNIST.labels();
      
      // find the loss and accuracy from the prediction
      prediction = model.forward(x);
      l = loss.compute(prediction, y);
      accuracy = metric.compute(prediction, y);

      System.out.println("Epoch " + epoch + " | accuracy: " + accuracy + " | loss: " + l.dump(0));

      for (DataContainer batch : loader) {
        x = batch.features();
        y = batch.labels();
        
        prediction = model.forward(x);
        l = loss.compute(prediction, y);
  
        optimizer.zeroGrad();
        l.backward();
        optimizer.step();
      }
    }
  }
}
