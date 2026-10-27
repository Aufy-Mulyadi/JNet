import com.aufy.jnet.jtensor.Tensor;

public class testTensor {
  public static void main(String[] args) {
    Tensor A = new Tensor(new double[]{
      1,1,  2,2,
      1,1,  2,2
      },
    2, 2, 2);
    Tensor B = new Tensor(new double[]{
      1,1,
      2,2
      },
    2, 2);

    Tensor C = A.contract(B, new int[]{1}, new int[]{0});

    System.out.println(C);
    // System.out.println(A.reduce.argmax(1));
    // System.out.println("====================");
  }


}
