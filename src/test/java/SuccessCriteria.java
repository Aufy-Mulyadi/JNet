
import java.util.Arrays;
import java.util.List;

import com.aufy.jnet.data.DataContainer;
import com.aufy.jnet.data.DataLoader;
import com.aufy.jnet.data.FileLoader;
import com.aufy.jnet.jtensor.Tensor;
import com.aufy.jnet.net.Module;

public class SuccessCriteria {

  public static void main(String[] args) {

    // TENSOR ENGINE

    // test1();
    // test2();
    test3();
    test4();
    test5();
    test6();
    test7();
    test8();
    test9();
    // test10();

    // AUTOMATIC DIFFERENTIATION

    test11();
    test12();
    test13();
    test14();

    // DATA PIPELINE

    // test15();
    // test16();
    // test17();
    // test18();

    // ADDITIONAL: NEURAL NETWORK COMPONENTS
    testIndividualLayers();
  }

  // =====================================================
  // TENSOR ENGINE
  // Success Criteria 1, 2 and 3
  // =====================================================

  // TEST 1: Create a tensor from a valid shape and data array
  public static void test1() {
    System.out.println("\n========== TEST 1: Valid Tensor Construction ==========");

    Tensor A = new Tensor(new double[] {
        1, 1,
        1, 1,
        2, 2,
        2, 2
    }, 2, 2, 2);

    System.out.println(A);
  }

  // TEST 2: Create a tensor with mismatching data length and shape
  public static void test2() {
    System.out.println("\n========== TEST 2: Mismatching Tensor Shape and Data ==========");

    try {
      Tensor A = new Tensor(new double[] {
          1, 1,
          2, 2
      }, 2, 3);

      System.out.println(A);
      System.out.println("Unexpected: Invalid tensor was accepted.");

    } catch (Exception e) {
      System.out.println("Expected error:");
      System.out.println(e);
    }
  }

  // TEST 3: Initialize a [2, 3, 4] tensor and inspect strides
  public static void test3() {
    System.out.println("\n========== TEST 3: Tensor Strides ==========");

    Tensor A = Tensor.init(2, 3, 4).ones();

    System.out.println("Tensor:");
    System.out.println(A);

    System.out.println("Strides:");
    System.out.println(Arrays.toString(A.internal.strides));
  }

  // TEST 4: Elementwise addition of two [2, 3, 4] tensors
  public static void test4() {
    System.out.println("\n========== TEST 4: Elementwise Tensor Addition ==========");

    Tensor A = Tensor.init(2, 3, 4).ones();
    Tensor B = Tensor.init(2, 3, 4).ones();

    System.out.println(A.add(B));
  }

  // TEST 5: Scalar addition of 2 to a [2, 3, 4] tensor
  public static void test5() {
    System.out.println("\n========== TEST 5: Scalar Addition ==========");

    Tensor A = Tensor.init(2, 3, 4).ones();

    System.out.println(A.add(2));
  }

  // TEST 6: Hadamard multiplication of two [2, 2] tensors
  public static void test6() {
    System.out.println("\n========== TEST 6: Hadamard Multiplication ==========");

    Tensor A = new Tensor(new double[] {
        1, 1,
        2, 2
    }, 2, 2);

    Tensor B = new Tensor(new double[] {
        2, 2,
        3, 3
    }, 2, 2);

    System.out.println(A.hadamard(B));
  }

  // TEST 7: In-place division by 2 using namespace routing
  public static void test7() {
    System.out.println("\n========== TEST 7: In-place Division ==========");

    Tensor A = new Tensor(new double[] {
        2, 2,
        3, 3
    }, 2, 2);

    System.out.println("Before division:");
    System.out.println(A);

    System.out.println("After division:");
    System.out.println(A.inplace.div(2));
  }

  // TEST 8: Matrix multiplication of two [2, 2] tensors
  public static void test8() {
    System.out.println("\n========== TEST 8: Matrix Multiplication ==========");

    Tensor A = new Tensor(new double[] {
        1, 1,
        2, 2
    }, 2, 2);

    Tensor B = new Tensor(new double[] {
        2, 2,
        3, 3
    }, 2, 2);

    System.out.println(A.matmul(B));
  }

  // TEST 9: Batched matrix multiplication
  public static void test9() {
    System.out.println("\n========== TEST 9: Batched Matrix Multiplication ==========");

    Tensor A = new Tensor(new double[] {
        1, 1
    }, 1, 2);

    Tensor B = new Tensor(new double[] {
        2, 2,
        3, 3,

        2, 2,
        3, 3,

        2, 2,
        3, 3
    }, 2, 2, 3);

    System.out.println(A.matmul(B));
  }

  // TEST 10: Multiply incompatible [2, 2] and [3, 2] matrices
  public static void test10() {
    System.out.println("\n========== TEST 10: Incompatible Matrix Multiplication ==========");

    Tensor A = new Tensor(new double[] {
        1, 1,
        2, 2
    }, 2, 2);

    Tensor B = new Tensor(new double[] {
        2, 2,
        3, 3,
        4, 4
    }, 3, 2);

    try {
      System.out.println(A.matmul(B));
      System.out.println("Unexpected: Incompatible matrices were accepted.");

    } catch (Exception e) {
      System.out.println("Expected dimension mismatch error:");
      System.out.println(e);
    }
  }

  // =====================================================
  // AUTOMATIC DIFFERENTIATION
  // Success Criteria 4 and 5
  // =====================================================

  // TEST 11: Compute gradient of y = 2x
  public static void test11() {
    System.out.println("\n========== TEST 11: Gradient of y = 2x ==========");

    Tensor A = new Tensor(new double[] {
        1, 1
    }, 2, 1).requiresGrad();

    Tensor B = A.mul(2);

    B.backward();

    System.out.println("Gradient:");
    System.out.println(A.grad());

    // Expected gradient: [2, 2]
  }

  // TEST 12: Compute gradient of y = x^2
  public static void test12() {
    System.out.println("\n========== TEST 12: Gradient of y = x^2 ==========");

    Tensor A = new Tensor(new double[] {
        2, 3
    }, 2, 1).requiresGrad();

    Tensor B = A.pow(2);

    B.backward();

    System.out.println("Gradient:");
    System.out.println(A.grad());

    // Expected gradient: [4, 6]
  }

  // TEST 13: Compute gradient through chained operations
  // f(x) = 2x
  // g(x) = x^2
  // h(x) = x + 1
  // D = f(g(h(x)))
  public static void test13() {
    System.out.println("\n========== TEST 13: Chained Computational Graph Gradient ==========");

    Tensor A = new Tensor(new double[] {
        2, 3
    }, 2, 1).requiresGrad();

    Tensor B = A.add(1); // h(x) = x + 1
    Tensor C = B.pow(2); // g(x) = x^2
    Tensor D = C.mul(2); // f(x) = 2x

    D.backward();

    System.out.println("Final gradient:");
    System.out.println(A.grad());

    // f(g(h(x))) = 2(x + 1)^2
    // Derivative = 4(x + 1)
    // Expected gradient: [12, 16]
  }

  // TEST 14: Call backward() on a non-differentiable tensor
  public static void test14() {
    System.out.println("\n========== TEST 14: Backward on Non-Differentiable Tensor ==========");

    Tensor A = new Tensor(new double[] {
        2, 3
    }, 2, 1);

    Tensor B = A.add(1);
    Tensor C = B.pow(2);
    Tensor D = C.mul(2);

    try {
      D.backward();

      System.out.println("Backward completed.");
      System.out.println("Gradient:");
      System.out.println(A.grad());

    } catch (Exception e) {
      System.out.println("Expected non-differentiable tensor error:");
      System.out.println(e);
    }
  }

  // =====================================================
  // DATA PIPELINE
  // Success Criterion 6
  // =====================================================

  // TEST 15: Load 5x5 CSV with features [2, 2] and labels [1]
  public static void test15() {
    System.out.println("\n========== TEST 15: CSV Loading with Feature Shape [2, 2] ==========");

    DataContainer testCSV = FileLoader.loadCSV(
        "src/test/resources/5x5.csv",
        new int[] { 0 },
        new int[] { 1 },
        new int[] { 2, 2 });

    Tensor feature = testCSV.features();
    Tensor label = testCSV.labels();

    System.out.println("Features:");
    System.out.println(feature);

    System.out.println("Labels:");
    System.out.println(label);
  }

  // TEST 16: Load 5x5 CSV with features [3, 3] and labels [1]
  public static void test16() {
    System.out.println("\n========== TEST 16: CSV Loading with Feature Shape [3, 3] ==========");

    DataContainer testCSV = FileLoader.loadCSV(
        "src/test/resources/5x5.csv",
        new int[] { 0 },
        new int[] { 1 },
        new int[] { 3, 3 });

    Tensor feature = testCSV.features();
    Tensor label = testCSV.labels();

    System.out.println("Features:");
    System.out.println(feature);

    System.out.println("Labels:");
    System.out.println(label);
  }

  // TEST 17: Load 100-row CSV using DataLoader with batch size 16
  public static void test17() {
    System.out.println("\n========== TEST 17: DataLoader with 100 Rows and Batch Size 16 ==========");

    DataContainer testCSV = FileLoader.loadCSV(
        "src/test/resources/100x10.csv",
        new int[] { 0 },
        new int[] { 1 },
        new int[] { 3, 3 });

    DataLoader loader = new DataLoader(testCSV, 16, true);

    int batchCount = 0;

    for (DataContainer batch : loader) {
      batchCount++;

      System.out.println("\nBatch " + batchCount);

      System.out.println("Features:");
      System.out.println(batch.features());

      System.out.println("Labels:");
      System.out.println(batch.labels());
    }

    System.out.println("\nTotal batches: " + batchCount);

    // Expected: 7 batches
    // Six batches of 16 and one batch of 4.
  }

  // TEST 18: Load a CSV containing strings or corrupted data
  public static void test18() {
    System.out.println("\n========== TEST 18: Corrupted CSV Input ==========");

    try {
      DataContainer testCSV = FileLoader.loadCSV(
          "src/test/resources/Corrupted5x5.csv",
          new int[] { 0 },
          new int[] { 1 },
          new int[] { 2, 2 });

      System.out.println("Features:");
      System.out.println(testCSV.features());

      System.out.println("Labels:");
      System.out.println(testCSV.labels());

      System.out.println("CSV loaded without an exception.");

    } catch (Exception e) {
      System.out.println("CSV loading failed as expected:");
      System.out.println(e);
    }
  }

  // =====================================================
  // NEURAL NETWORK COMPONENTS
  // Success Criteria 7 and 8
  // =====================================================

  public static void testIndividualLayers() {
    System.out.println("\n========== ADDITIONAL TEST: Individual Layer ==========");

    CustomLayer testLayer = new CustomLayer(3, 2, 10);

    Tensor input = Tensor.init(2, 3).ones();

    Tensor out = testLayer.forward(input);

    System.out.println("Layer output:");
    System.out.println(out);

    out.backward();

    System.out.println("Input gradient:");
    System.out.println(input.grad());

    System.out.println("Layer parameters:");
    System.out.println(testLayer.parameters());
  }

}

class CustomLayer extends Module {
  private final Tensor weight;
  private final Tensor bias;
  private final double shift;

  public CustomLayer(int incoming, int outgoing, double constantShift) {
    weight = Tensor.init(outgoing, incoming).normal(0, 1).requiresGrad();
    bias = Tensor.init(1, outgoing).zeros().requiresGrad();
    shift = constantShift;
  }

  @Override
  public Tensor forward(Tensor x) {
    return x.matmul(weight.transpose()).add(bias).add(shift);
  }

  @Override
  public List<Tensor> parameters() {
    return List.of(weight, bias);
  }
}
