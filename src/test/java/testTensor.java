import com.aufy.jnet.jtensor.Tensor;

public class testTensor {
  public static void main(String[] args) {
    Tensor A = Tensor.init(2,5).arrange();
    Tensor B = Tensor.init(5,2).arrange();
    // Tensor A = new Tensor(new double[]{
    //   -0.0149, -0.0168, -0.0110, -0.0159, -0.0162, -0.0148, -0.0134, -0.0183, -0.0146, -0.0091,
    //   -0.0162, -0.0192, -0.0215, -0.0023, -0.0181, -0.0006, -0.0107, -0.0100, -0.0003, -0.0057,
    //   -0.0166, -0.0169, -0.0128, -0.0148, -0.0157, -0.0129, -0.0118, -0.0166, -0.0165, -0.0154
    //   },
    // 3, 10);

    // Tensor B = new Tensor(new double[]{1, 2, 3, 4, 5}, 1,5);
    // double[] example = new double[] {-0.0166, -0.0169, -0.0128, -0.0148, -0.0157, -0.0129, -0.0118, -0.0166, -0.0165, -0.0154};
    Tensor C = A.contract(B, new int[]{1}, new int[]{0});

    System.out.println(C);
    // System.out.println(A.reduce.argmax(1));
    // System.out.println("====================");
  }


}
