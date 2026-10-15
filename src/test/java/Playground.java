import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.arrayops.Tools;
import com.aufy.jnet.core.tensor.core.backend.compute.Memory;
import com.aufy.jnet.core.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.core.tensor.core.backend.util.Visual;

public class Playground {

  public static void main(String[] OwO) {
    int[] shapeA = new int[] {4,5};
    int[] shapeB = new int[] {5,4};

    double[] dataA = Tools.arrangeDouble(Reductions.prod(shapeA));
    double[] dataB = Tools.arrangeDouble(Reductions.prod(shapeB));

    int[] axesA = new int[] {shapeA.length - 1};
    int[] axesB = new int[] {shapeA.length - 2};

    int[] resShape = Shaping.calculateResShape(shapeA, shapeB, axesA, axesB);

    double[] res = new double[Reductions.prod(resShape)];
    double startTime = System.nanoTime();
    for (int i = 0; i < 10000; i++) {
      res = 
      bmm(
        dataA, 
        Memory.calculateStrides(shapeA), 
        dataB, 
        Memory.calculateStrides(shapeB), 
        shapeA,
        axesA,
        shapeB, 
        axesB,
        resShape
      );


      // contractBMM(
      //   dataA, 
      //   Memory.calculateStrides(shapeA), 
      //   dataB, 
      //   Memory.calculateStrides(shapeB), 
      //   shapeA,
      //   shapeB,
      //   resShape
      // );
    }
    double durationMs = (System.nanoTime() - startTime) / 1_000_000.0;
    System.out.println("Execution time (milliseconds): " + durationMs);
    

    System.out.println(Visual.print(dataA, shapeA, Memory.calculateStrides(shapeA), 0, 0, 0));
    System.out.println(Visual.print(dataB, shapeB, Memory.calculateStrides(shapeB), 0, 0, 0));
    System.out.println("==========================");
    System.out.println(Visual.print(res, resShape, Memory.calculateStrides(resShape), 0, 0, 0));
  }

  /**
   * Performs a tesor contraction along specified axes. Tensor contaction is basically sum_[indexes](A[indexes] * B[indexes]) = C[leftover dims].
   * This is a more advanced version of matrix multiplication.
   * 
   * @param dataA the data of the first tensor.
   * @param stridesA the strides of the first tensor.
   * @param dataB the data of the second tensor.
   * @param stridesB the strides of the second tensor.
   * @param shapeA the shape of the first tensor.
   * @param axesA the axes of the first tensor.
   * @param shapeB the shape of the second tensor.
   * @param axesB the axes of the second tensor.
   * @param resShape the shape of the result tensor.
   * @return the data of the result tensor.
   */
  public static double[] bmm(double[] dataA, int[] stridesA, double[] dataB, int[] stridesB, int[] shapeA, int[] axesA, int[] shapeB, int[] axesB, int[] resShape) {
    double[] resData = new double[Reductions.prod(resShape)];

    // extract the subshape according to the axes
    int[] subShape = Shaping.reorder(shapeA, axesA);
    int contractVolume = Reductions.prod(subShape); // final result volume
    
    // speed thigs up by precomputing the offsets
    int[] OffsetA = Shaping.findOffset(subShape, axesA, stridesA);
    int[] OffsetB = Shaping.findOffset(subShape, axesB, stridesB);

    int[] resCoords = new int[resShape.length];
    int resIdx = 0;

    int[] emptyA = new int[axesA.length];
    int[] emptyB = new int[axesB.length];

    do {
      int baseOffsetA = Memory.mapToOffset(resCoords, emptyA, axesA, shapeA, stridesA, true);
      int baseOffsetB = Memory.mapToOffset(resCoords, emptyB, axesB, shapeB, stridesB, false);

      double sum = 0;
      for (int k = 0; k < contractVolume; k++) sum += dataA[baseOffsetA + OffsetA[k]] * dataB[baseOffsetB + OffsetB[k]];
      
      resData[resIdx] = sum;
      resIdx++;
    } while (
      Memory.nextCoordinate(resCoords, resShape)
    );

    return resData;
  }

  public static double[] contractBMM(
    double[] dataA, int[] stridesA, 
    double[] dataB, int[] stridesB, 
    int[] shapeA, int[] shapeB, int[] resShape
) {
    // 1. Extract dimensions
    int N_A = shapeA.length;
    int N_B = shapeB.length;
    
    int M = shapeA[N_A - 2];
    int K = shapeA[N_A - 1]; // contracting axis n-1
    int N = shapeB[N_B - 1]; // remaining axis n-1 (since n-2 is K)

    // Compute total batch size by multiplying all leading dimensions
    int batchSize = 1;
    for (int i = 0; i < N_A - 2; i++) {
        batchSize *= shapeA[i];
    }

    double[] resData = new double[batchSize * M * N];

    // 2. Extract crucial strides for direct pointer math
    int strideAM = stridesA[N_A - 2];
    int strideAK = stridesA[N_A - 1];
    int strideBK = stridesB[N_B - 2];
    int strideBN = stridesB[N_B - 1];

    // Precompute batch strides for A and B
    int[] batchStridesA = precomputeBatchStrides(shapeA, stridesA);
    int[] batchStridesB = precomputeBatchStrides(shapeB, stridesB);

    // 3. Cache-Conscious Blocked Loop (Tiling)
    int BLOCK = 64; // Adjust based on L1/L2 cache size

    for (int b = 0; b < batchSize; b++) {
        int baseA_batch = batchStridesA[b];
        int baseB_batch = batchStridesB[b];
        int baseRes_batch = b * M * N;

        // Blocked GEMM loops (Loop Tiling)
        for (int kk = 0; kk < K; kk += BLOCK) {
            int kEnd = Math.min(kk + BLOCK, K);
            
            for (int ii = 0; ii < M; ii += BLOCK) {
                int iEnd = Math.min(ii + BLOCK, M);
                
                for (int jj = 0; jj < N; jj += BLOCK) {
                    int jEnd = Math.min(jj + BLOCK, N);

                    // Inner micro-kernels
                    for (int i = ii; i < iEnd; i++) {
                        int rowOffsetA = baseA_batch + i * strideAM;
                        int rowOffsetRes = baseRes_batch + i * N;

                        for (int k = kk; k < kEnd; k++) {
                            int idxA = rowOffsetA + k * strideAK;
                            double valA = dataA[idxA];
                            int rowOffsetB = baseB_batch + k * strideBK;

                            // Innermost loop: Sequential memory access over N
                            for (int j = jj; j < jEnd; j++) {
                                resData[rowOffsetRes + j] += valA * dataB[rowOffsetB + j * strideBN];
                            }
                        }
                    }
                }
            }
        }
    }
    return resData;
}

private static int[] precomputeBatchStrides(int[] shape, int[] strides) {
    // 1. Determine total number of batch matrices
    int batchSize = 1;
    for (int i = 0; i < shape.length - 2; i++) {
        batchSize *= shape[i];
    }
    
    int[] batchOffsets = new int[batchSize];
    int[] currentCoords = new int[shape.length - 2]; // Tracks leading dimensions only
    
    // 2. Compute flat memory offsets for every batch combo
    for (int b = 0; b < batchSize; b++) {
        int offset = 0;
        for (int i = 0; i < currentCoords.length; i++) {
            offset += currentCoords[i] * strides[i];
        }
        batchOffsets[b] = offset;
        
        // Manual coordinate increment for the leading batch dimensions
        for (int i = currentCoords.length - 1; i >= 0; i--) {
            currentCoords[i]++;
            if (currentCoords[i] < shape[i]) {
                break; // Successfully incremented
            }
            currentCoords[i] = 0; // Carry over to the next dimension
        }
    }
    return batchOffsets;
}

}
