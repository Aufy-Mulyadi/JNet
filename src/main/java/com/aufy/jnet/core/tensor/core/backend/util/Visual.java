package com.aufy.jnet.core.tensor.core.backend.util;

import java.util.Arrays;

import com.aufy.jnet.core.tensor.core.implementations.CoreTensor;

public class Visual {
  
  public static void trace(CoreTensor tensor) {
    printGraph(tensor, "", true, true);
  }
  
  private static void printGraph(CoreTensor node, String indent, boolean isLast, boolean firstLayer) {
    String operation = (node.derivative != null) ? " [Product]" : " [Leaf]";
    String shape = Arrays.toString(node.shape);
    
    System.out.print(indent);
    System.out.print(isLast ? "└─ " : "├─ ");
    
    if (firstLayer) {
      System.out.println("\r#  Tensor" + shape + " [ROOT]");
    } else {
      System.out.println("Tensor" + shape + operation);
    }

    // BFS over parents
    if (node.parents != null) {
      for (int i = 0; i < node.parents.size(); i++) {
        boolean lastChild = (i == node.parents.size() - 1);
        printGraph(node.parents.get(i), indent + (isLast ? "   " : "│  "), lastChild, false);
      }
    }
  }

  public static String print(double[] data, int[] shape, int[] strides, int rank, int offset, int shift) {
    int width = 0;
    for (double d : data) {
      width = Math.max(width, String.format("%.4f", d).length());
    }

    if (shape.length == 0) return String.format("%.4f", data[0]);
    
    StringBuilder sb = new StringBuilder("");
    
    if (rank == 0) {
      sb.append(" ".repeat(shift));
    }

    sb.append("[");
    
    int edgeSize = 3; // keep 3 on the edges
    int trigger = 10; // if a dimension is more than 10, truncate display
    int i = 0;
    while (i < shape[rank]) {
      int currentOffset = offset + (i * strides[rank]);

      if (rank == shape.length - 1) { // base case
        sb.append(String.format("%" + width + ".4f", data[currentOffset]).replace(',','.'));
      } else {
        sb.append(print(data, shape, strides, rank + 1, currentOffset, shift));
      }
      
      if (shape[rank] > trigger && i == edgeSize - 1) { // skip case
        if (rank == shape.length - 1) { // print the column gaps ([a ... b])
          sb.append(" ... ");

        } else { // print the sheet gaps ([a] ... [b])
          int newlineCount = Math.max(1, shape.length - rank - 1);
          sb.append("\n".repeat(newlineCount));
          sb.append(" ".repeat(shift + rank + 1)).append("...\n");
          sb.append(" ".repeat(shift + rank + 1));
        }

        i = shape[rank] - edgeSize;
      } else { // normal case
        if (i < shape[rank] - 1) {
          if (rank == shape.length - 1) { // print the column gaps ([a b c])
            sb.append(" ");

          } else { // print the sheet gaps ([a] [b] [c])
            int newlineCount = Math.max(1, shape.length - rank - 1);
            sb.append("\n".repeat(newlineCount));
            sb.append(" ".repeat(shift));
            sb.append(" ".repeat(rank + 1));
          }
        }

        i++;
      }

    }
    sb.append("]");
    return sb.toString();
  }
}