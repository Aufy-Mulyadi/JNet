package com.aufy.jnet.core.tensor.core.implementations;
import java.util.Arrays;

import com.aufy.jnet.core.tensor.core.backend.compute.Memory;
import com.aufy.jnet.core.tensor.core.backend.util.Visual;

public class RawTensor {
  public double[] data;
  public int[] strides;
  public int[] shape;
  public int rank;
  public int size;
  
  public RawTensor(double[] data, int... shape) {
    this.data = data;
    this.shape = shape;
    this.strides = Memory.calculateStrides(shape);
    this.rank = this.shape.length;
    this.size = this.data.length;
  }

  // ########################################################################################################### //
  //                                                  UTILITY                                                    //
  // ########################################################################################################### //

  public RawTensor duplicate() {return duplicate(this);}
  public static RawTensor duplicate(RawTensor tensor) {
    RawTensor out = new RawTensor(tensor.data.clone(), tensor.shape);
    
    // copy RawTensor attribute
    out.strides = tensor.strides.clone();

    return out;
  }

  public double[] dump() {return this.data;}

  public double get(int... indices) {
    return this.data[Memory.getIndex(this.strides, indices)];
  }

  @Override
  public String toString() {
    if (this.data == null || this.shape == null || this.data.length == 0) return "RawTensor[null]()";
    
    String prefix = "RawTensor" + Arrays.toString(this.shape) + "(\n";
    String content = Visual.print(this.data, this.shape, this.strides, 0, 0, 2);

    return prefix + content + "\n)";
  }
}