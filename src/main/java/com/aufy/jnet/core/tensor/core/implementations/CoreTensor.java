package com.aufy.jnet.core.tensor.core.implementations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;

import com.aufy.jnet.core.backend.exceptions.internal.Data;
import com.aufy.jnet.core.tensor.core.backend.util.Visual;
import com.aufy.jnet.core.tensor.raw.init.RawTensorGenerator;

public class CoreTensor {
  public static boolean verbose = false;
  public RawTensor core;
  
  public int rank;
  public int size;
  public int[] shape;
  public boolean requiresGrad;
  
  public List<CoreTensor> cachedGraphOrder;
  public List<CoreTensor> parents;
  public RawTensor grad; // stores d[end]/d[this]
  public Consumer<RawTensor> derivative; // rawtensor -> void that gives the [grad] values to its parents
  
  public CoreTensor(double[] data, int... shape) {
    this(new RawTensor(data, shape));
  }

  public CoreTensor(RawTensor core) {
    this.core = core;
    this.requiresGrad = false;
    this.shape = core.shape;
    this.size = core.size;
    this.rank = core.rank;
  }

  // ########################################################################################################### //
  //                                                  UTILITY                                                    //
  // ########################################################################################################### //
  
  public CoreTensor duplicate() {return duplicate(this);}
  public static CoreTensor duplicate(CoreTensor tensor) {
    CoreTensor out = new CoreTensor(tensor.core.duplicate());

    // copy CoreTensor attributes
    out.requiresGrad = tensor.requiresGrad;
    out.grad = tensor.grad;
    out.parents = tensor.parents;
    out.derivative = tensor.derivative;

    return out;
  }

  public double[] dump() {
    return this.core.dump();
  }

  public CoreTensor detach() {return detach(this);}
  public static CoreTensor detach(CoreTensor tensor) {
    tensor.requiresGrad = false;
    tensor.parents = null;
    tensor.derivative = null;

    return tensor;
  }

  public double get(int... indices) {
    return this.core.get(indices);
  }

  @Override
  public String toString() {
    if (this.core.data == null || this.shape == null || this.core.data.length == 0) return "CoreTensor[null]()";

    String prefix = "CoreTensor" + Arrays.toString(this.shape) + "(\n";
    String content = Visual.print(this.core.data, this.shape, this.core.strides, 0, 0, 2);

    if (verbose) {
      return prefix + content + "\n\n" + " grad=" + this.requiresGrad + "\n" + " parents=" + this.parents.size() + "\n" + " derivative=" + this.derivative + "\n)";
    } else {
      return prefix + content + "\n)";
    }
  }

  // ########################################################################################################### //
  //                                              AUTOGRAD LOGIC                                                 //
  // ########################################################################################################### //

  public void zeroGrad() {CoreTensor.zeroGrad(this);}
  public static void zeroGrad(CoreTensor tensor) {
    if (tensor.grad != null) Arrays.fill(tensor.grad.data, 0);
  }

  public void accumulate(double[] incomingGrad) {
    if (!this.requiresGrad) return;
    if (this.grad == null) this.grad = RawTensorGenerator.zerosLike(this.core);

    Data.verifyEqualSizes("Data transfer", this.core.data, incomingGrad);
    
    for (int i = 0; i < this.size; i++) this.grad.data[i] += incomingGrad[i];
  }

  public void accumulate(RawTensor incomingGrad) {
    if (!this.requiresGrad) return;
    if (this.grad == null) this.grad = RawTensorGenerator.zerosLike(this.core);

    Data.verifyEqualSizes("Data transfer", this.core.data, incomingGrad.data);
    
    for (int i = 0; i < this.size; i++) this.grad.data[i] += incomingGrad.data[i];
  }

  public static List<CoreTensor> buildGraph(CoreTensor root) {
    List<CoreTensor> order = new ArrayList<>();
    Set<CoreTensor> visited = new HashSet<>();
    
    visit(root, visited, order);
    
    // reverse the order because we want to go from out --> in
    Collections.reverse(order);
    return order;
  }

  private static void visit(CoreTensor node, Set<CoreTensor> visited, List<CoreTensor> order) {
    if (node == null || visited.contains(node)) return;
    
    visited.add(node);
    if (node.parents != null) {
      for (CoreTensor parent : node.parents) {
        visit(parent, visited, order);
      }
    }
    order.add(node);
  }

  public void backward() {
    this.grad = RawTensorGenerator.onesLike(this.core);

    if (this.cachedGraphOrder == null) {
      this.cachedGraphOrder = buildGraph(this);
    }

    for (int i = 0; i < this.cachedGraphOrder.size(); i++) {
      CoreTensor node = this.cachedGraphOrder.get(i);

      if (node.derivative != null && node.grad != null) node.derivative.accept(node.grad);
    }
  }

}