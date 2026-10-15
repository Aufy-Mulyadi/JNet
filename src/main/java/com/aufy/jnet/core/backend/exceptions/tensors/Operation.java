package com.aufy.jnet.core.backend.exceptions.tensors;

import java.util.Arrays;

import com.aufy.jnet.core.backend.exceptions.Message;

/**
 * Utility class for validating structural and dimension requirements of tensor operations.
 */
public class Operation {
  /*
   * specific implementations go here; VIPs only
   */

  /**
   * Checks if two tensor shapes are compatible for matrix multiplication.
   *
   * @param shapeA the shape of the left tensor
   * @param shapeB the shape of the right tensor
   * @throws IllegalArgumentException if ranks are under 1 or inner dimensions mismatch
   */
  public static void verifyMatMul(int[] shapeA, int[] shapeB) throws IllegalArgumentException {
    if (shapeA.length < 1 || shapeB.length < 1) {
      Message.crash("tensor operation", "Incompatible rank", "matmul", "tensors must have a rank of at least 1");
    }

    //                     v           v
    // matmul  = A[..., i, j] * B[..., j, k] = C[..., i, k]
    int indexA = shapeA[shapeA.length - 1]; // last
    int indexB = (shapeB.length == 1) ? shapeB[0] : shapeB[shapeB.length - 2]; // 2nd dim if there is one, else remaining one

    if (indexA != indexB) {
      Message.crash("tensor operation", "Incompatible dimensions for matrix multiplication", "matmul", "index " + indexA + " from tensor A and index " + indexB + " from tensor B");
    }
  }

  /**
   * Checks if multiple shapes match exactly on all dimensions except the concatenation axis.
   *
   * @param axis the dimension index along which tensors are joined
   * @param shapes the collection of shapes to combine
   * @throws IllegalArgumentException if ranks vary or non-axis dimensions do not match
   */
  public static void verifyConcat(int axis, int[]... shapes) throws IllegalArgumentException {
    int rankRef = shapes[0].length;
    if (axis < 0 || axis >= rankRef) Message.crash("tensor operation", "Invalid axis for concatenation", "concatenation", "axis " + axis + " is out of bounds");

    for (int i = 1; i < shapes.length; i++) {
      if (shapes[i].length != rankRef) {
        Message.crash("tensor operation", "Mismatching rank at index " + i + " for concatenation", "concatenation", "expected " + rankRef + " but got " + shapes[i].length);
      }
      for (int d = 0; d < rankRef; d++) {
        if (d != axis && shapes[i][d] != shapes[0][d]) {
          Message.crash("tensor operation", "Dimension mismatch at axis " + d, "concatenation", shapes[0][d] + " vs " + shapes[i][d]);
        }
      }
    }
  }

  /**
   * Checks if all provided shapes are completely identical for stacking.
   *
   * @param shapes the collection of shapes to stack
   * @throws IllegalArgumentException if any shape differs from the first one
   */
  public static void verifyStack(int[]... shapes) {
    for (int i = 1; i < shapes.length; i++) {
      if (!Arrays.equals(shapes[0], shapes[i])) {
        Message.crash("tensor operation", "Stack requires identical shapes", "stack", "Mismatch at index " + i + ": " + Arrays.toString(shapes[0]) + " vs " + Arrays.toString(shapes[i]));
      }
    }
  }

  /**
   * Checks if two shapes are compatible for element-wise broadcasting rules.
   *
   * @param operationName the name of the calling operation
   * @param shapeA the shape of the first tensor
   * @param shapeB the shape of the second tensor
   * @throws IllegalArgumentException if dimensions cannot stretch to match each other
   */
  public static void verifyBroadcast(String operationName, int[] shapeA, int[] shapeB) throws IllegalArgumentException {
    for (int i = 0; i < shapeA.length; i++) {
      if (shapeA[i] != shapeB[i] && (shapeA[i] != 1 && shapeB[i] != 1)) {
        Message.crash("tensor operation", "Unable to broadcast shapes", operationName, "broadcasting only works if either dimensions are 1 or match, but axis " + i + " has a dimension of " + shapeA[i] + " in tensor A (" + Arrays.toString(shapeA) + ") while the same axis corresponds to a dimension of " + shapeB[i] + " in tensor B (" + Arrays.toString(shapeB) + ")");
      }
    }
  }

}
