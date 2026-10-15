package com.aufy.jnet.jtensor;

import java.util.Arrays;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.arrayops.Tools;
import com.aufy.jnet.core.backend.exceptions.Message;
import com.aufy.jnet.core.backend.exceptions.internal.Data;
import com.aufy.jnet.core.backend.exceptions.tensors.Argument;
import com.aufy.jnet.core.backend.exceptions.tensors.Axes;
import com.aufy.jnet.core.backend.exceptions.tensors.Operation;
import com.aufy.jnet.core.tensor.core.backend.util.Visual;
import com.aufy.jnet.core.tensor.core.implementations.CoreTensor;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;
import com.aufy.jnet.core.tensor.graph.main.BinaryOps;
import com.aufy.jnet.core.tensor.graph.main.ShapeOps;

/**
 * The main Tensor class of JNet. This class is a wrapper around the CoreTensor
 * class but it is advised to use the CoreTensor class directly because
 * there are no built in methods and there are no safety checks (unlike this
 * specialized class).
 */
public class Tensor {
  public Reduce reduce;
  public Custom custom;
  public Ops ops;
  public Util util;
  public Stats stats;
  public Internal internal;
  public Logic logic;

  public com.aufy.jnet.jtensor.inplace.Inplace inplace;
  public com.aufy.jnet.jtensor.inplace.Fill fill_;
  public com.aufy.jnet.jtensor.inplace.Ops ops_;
  public com.aufy.jnet.jtensor.inplace.Util util_;

  protected final CoreTensor core;

  protected int[] shape;
  protected int rank;
  protected int size; 

  public static boolean verbose = false;
  public boolean requiresGrad;

  public Tensor(double[] data, int... shape) {
    if (data == null) Message.crash("initialization", "Null data", null, "data cannot be null");
    if (data.length == 0) Message.crash("initialization", "Empty data", null, "data cannot be empty");
    if (shape == null) Message.crash("initialization", "Null shape", null, "shape cannot be null");
    if (shape.length == 0) Message.crash("initialization", "Empty shape", null, "shape cannot be empty");
    if (Reductions.prod(shape) != data.length) Message.crash("initialization", "Bad shape", null, "shape " + Arrays.toString(shape) + " does not match data size " + data.length);

    this(new CoreTensor(data, shape));
  }
  
  protected Tensor(CoreTensor tensor) {
    this.shape = tensor.shape;
    this.rank = tensor.rank;
    this.size = tensor.size;
    this.requiresGrad = tensor.requiresGrad;
    this.core = tensor;
    
    this.reduce = new Reduce(this);
    this.custom = new Custom(this);
    this.ops = new Ops(this);
    this.util = new Util(this);
    this.stats = new Stats(this);
    this.internal = new Internal(this);
    this.logic = new Logic(this);

    this.inplace = new com.aufy.jnet.jtensor.inplace.Inplace(this);
    this.fill_ = new com.aufy.jnet.jtensor.inplace.Fill(this);
    this.ops_ = new com.aufy.jnet.jtensor.inplace.Ops(this);
    this.util_ = new com.aufy.jnet.jtensor.inplace.Util(this);
  }
  
  protected Tensor(RawTensor tensor) {
    this.shape = tensor.shape;
    this.rank = tensor.rank;
    this.size = tensor.size;
    this.requiresGrad = false;
    this.core = new CoreTensor(tensor);

    this.reduce = new Reduce(this);
    this.custom = new Custom(this);
    this.ops = new Ops(this);
    this.util = new Util(this);
    this.stats = new Stats(this);
    this.internal = new Internal(this);
    this.logic = new Logic(this);

    this.inplace = new com.aufy.jnet.jtensor.inplace.Inplace(this);
    this.fill_ = new com.aufy.jnet.jtensor.inplace.Fill(this);
    this.ops_ = new com.aufy.jnet.jtensor.inplace.Ops(this);
    this.util_ = new com.aufy.jnet.jtensor.inplace.Util(this);
  }

  // ###########################################################################################################
  // UTILITY //
  // ###########################################################################################################

  @Override
  public String toString() {
    if (this.core.dump() == null || this.shape == null || this.core.dump().length == 0)
      return "UninitializedTensor[]()";

    String prefix = "Tensor" + Arrays.toString(this.shape) + "(\n";
    String content = Visual.print(this.core.dump(), this.shape, this.core.core.strides, 0, 0, 2);
    String suffix = " grad=" + this.requiresGrad;

    if (verbose) {
      return prefix + content + "\n\n" + suffix + "\n)";
    } else {
      return prefix + content + "\n)";
      // return prefix + "\n)";
    }
  }

  public Tensor duplicate() {
    return new Tensor(this.core.duplicate());
  }

  public static Tensor duplicate(Tensor tensor) {
    return new Tensor(tensor.core.duplicate());
  }

  /**
   * Dumps all of the tensor's raw heap.
   * 
   * @return a deep copy of the tensor's raw heap.
   */
  public double[] dump() {
    return dump(this);
  }

  /**
   * Dumps a specific element from a tensor's raw heap.
   * 
   * @param tensor the tensor to dump from.
   * @param index the index of the element to dump.
   * @return the element at the specified index.
   */
  public double dump(int index) {
    return dump(this)[index];
  }

  /**
   * Dumps all of the tensor's raw heap.
   * 
   * @param tensor the tensor to dump from.
   * @return a deep copy of the tensor's raw heap.
   */
  public static double[] dump(Tensor tensor) { // safe dump
    Data.verifyData(tensor.core.dump());
    return tensor.core.dump().clone();
  }
  
  /**
   * Dumps a specific element from a tensor's raw heap.
   * 
   * @param tensor the tensor to dump from.
   * @param index the index of the element to dump.
   * @return the element at the specified index.
   */
  public static double dump(Tensor tensor, int index) {
    return dump(tensor)[index];
  }

  public int[] getShape() {return this.shape.clone();}
  public int getShape(int index) {return this.shape[index];}
  public int getRank() {return this.rank;}
  public int getSize() {return this.size;}

  // ###########################################################################################################
  // AUTOGRAD //
  // ###########################################################################################################

  /**
   * Computes the gradients of this tensor with respect to the graph leaves.
   * Runs the backward pass starting from this tensor.
   */
  public void backward() {
    backward(this);
  }

  /**
   * Computes the gradients of a tensor with respect to the graph leaves.
   * Runs the backward pass starting from the specified tensor.
   *
   * @param tensor the tensor to run the backward pass from.
   */
  public static void backward(Tensor tensor) {
    if (!tensor.requiresGrad) Message.crash("autograd", "cannot compute gradients", "backward", "tensor is not configured to track gradients");
    tensor.core.backward();
  }

  /**
   * Retrieves the accumulated gradient tensor for this tensor instance.
   *
   * @return a new tensor containing the tracked gradient values.
   */
  public Tensor grad() {
    return grad(this);
  }

  /**
   * Retrieves the accumulated gradient tensor for the specified tensor.
   *
   * @param tensor the source tensor to extract the gradient from.
   * @return a new tensor containing the tracked gradient values.
   */
  public static Tensor grad(Tensor tensor) {
    if (!tensor.requiresGrad) Message.crash("autograd", "cannot compute gradients", "backward", "tensor is not configured to track gradients");
    if (tensor.core.grad == null) Message.crash("autograd", "cannot compute gradients", "backward", "tensor is not configured to track gradients");

    return new Tensor(tensor.core.grad);
  }

  /**
   * Disables gradient tracking for this tensor instance.
   */
  public void noGrad() {
    this.core.requiresGrad = false;
  }

  /**
   * Disables gradient tracking for the specified core tensor.
   *
   * @param tensor the target core tensor structure to modify
   */
  public static void noGrad(Tensor tensor) {
    if (tensor.core.grad == null) Message.crash("autograd", "null gradients", "backward", "tensor does not have gradients");
    if (tensor.core.grad == null) Message.crash("autograd", "cannot clear gradients", "backward", "tensor is not configured to track gradients");
    tensor.noGrad();
  }

  /**
   * Detaches this tensor from the current computational graph.
   * The returned tensor will not track further operations for autograd.
   *
   * @return a new isolated tensor detached from the graph history.
   */
  public Tensor detach() {
    return detach(this);
  }

  /**
   * Detaches a specified tensor from its computational graph.
   * The returned tensor will not track further operations for autograd.
   *
   * @param tensor the source tensor to isolate.
   * @return a new isolated tensor detached from the graph history.
   */
  public static Tensor detach(Tensor tensor) {
    tensor.core.detach();
    return tensor;
  }

  /**
   * Resets the accumulated gradients of this tensor instance back to zero.
   */
  public void zeroGrad() {
    zeroGrad(this);
  }

  /**
   * Resets the accumulated gradients of the specified tensor back to zero.
   *
   * @param tensor the target tensor whose gradients should be cleared.
   */
  public static void zeroGrad(Tensor tensor) {
    tensor.core.zeroGrad();
  }

  /**
   * Enables autograd tracking flags for this tensor instance.
   * Makes this tensor and future operations eligible for backward graph passes.
   * 
   * @return this tensor with autograd configuration enabled.
   */
  public Tensor requiresGrad() {
    return requiresGrad(this);
  }

  /**
   * Enables autograd tracking flags for the specified tensor.
   * Makes the tensor and future operations eligible for backward graph passes.
   *
   * @param tensor the target tensor to configure.
   * @return the same tensor instance with autograd configuration enabled.
   */
  public static Tensor requiresGrad(Tensor tensor) {
    tensor.requiresGrad = true;
    tensor.core.requiresGrad = true;
    return tensor;
  }
  
  // ###########################################################################################################
  // INTERNAL CALLS //
  // ###########################################################################################################

  /**
   * Initializes a new UninitializedTensor with the specified shape. To get a proper Tensor output, call the various
   * generating functions inside the UninitializedTensor.
   * 
   * @param shape the desired dimensions for the new tensor.
   * @return a new UninitializedTensor (if plainly called) or a new Tensor (if called initializing methods).
   */
  public static Init init(int... shape) {
    return new Init(shape);
  }

  /**
   * Initializes a new UninitializedTensor with the specified shape of another tensor. To get a proper Tensor output, call the various
   * generating functions inside the UninitializedTensor.
   * 
   * @param tensor the desired dimensions for the new tensor (matched to the shape).
   * @return a new UninitializedTensor (if plainly called) or a new Tensor (if called initializing methods).
   */
  public static Init init(Tensor tensor) {
    return new Init(tensor);
  }

  /**
   * Initializes a new UninitializedTensor with the same shape as this tensor. To get a proper Tensor output, call the various
   * generating functions inside the UninitializedTensor.
   * 
   * @return a new UninitializedTensor (if plainly called) or a new Tensor (if called initializing methods).
   */
  public Init init() {
    return new Init(this.shape);
  }

  /**
   * Creates a new Statistics object for this tensor. To get a proper Tensor output, call the various
   * generating functions inside the reference tensor.
   * 
   * @param axes the dimensions to reduce.
   * @return a reference tensor to this tensor with the specified axes reduced.
   */
  public Stats stats(int... axes) {
    return new Stats(this, axes);
  }

  /**
   * Creates a new Statistics object for this tensor along all axes. To get a proper Tensor output, call the various
   * generating functions inside the reference tensor.
   * 
   * @return a reference tensor to this tensor with the specified axes reduced.
   */
  public Stats stats() {
    return new Stats(this, Tools.arrange(this.rank));
  }
  
  // ###########################################################################################################
  // SPECIFIC OPERATIONS //
  // ###########################################################################################################

  // BINARY OPERATIONS ---------------------------------

  /**
   * Contracts this tensor with another tensor along the specified axes.
   *
   * @param tensorB the tensor to contract with.
   * @param axesA the axes of contraction for this tensor.
   * @param axesB the axes of contraction for the second tensor.
   * @return a new tensor resulting from the contraction.
   */
  public Tensor contract(Tensor tensorB, int[] axesA, int[] axesB) {
    return contract(this, tensorB, axesA, axesB);
  }

  /**
   * Contracts two specified tensors along their respective axes locations. Tensor contraction is a generalization of matrix multiplication; resulting tensors are automatically squeezed to remove single-dimensional entries.
   *
   * @param tensorA the first tensor.
   * @param tensorB the second tensor.
   * @param axesA the axes of contraction for tensorA.
   * @param axesB the axes of contraction for tensorB.
   * @return a new tensor resulting from the contraction.
   */
  public static Tensor contract(Tensor tensorA, Tensor tensorB, int[] axesA, int[] axesB) {
    if (tensorA == null || tensorB == null) Message.crash("internal error", "null tensors in tensor contraction", "tensor contraction", "tensors cannot be null");
    Axes.verifyAxes("tensor contraction", tensorA.rank, axesA);
    Axes.verifyAxes("tensor contraction", tensorB.rank, axesB);

    return new Tensor(BinaryOps.contract(tensorA.core, tensorB.core, axesA, axesB));
  }

  /**
   * Calculates the outer product of this tensor with another tensor to expand dimensions.
   *
   * @param tensorB the tensor to contract with.
   * @return a new tensor resulting from the product.
   */
  public Tensor outer(Tensor tensorB) {
    return outer(this, tensorB);
  }

  /**
   * Calculates the outer product of this tensor with another tensor to expand dimensions.
   *
   * @param tensorA the first tensor.
   * @param tensorB the second tensor.
   * @return a new tensor resulting from the product.
   */
  public static Tensor outer(Tensor tensorA, Tensor tensorB) {
    return new Tensor(BinaryOps.contract(tensorA.core, tensorB.core, new int[0], new int[0]));
  }
  
  /**
   * Adds another tensor to this tensor elementwise.
   *
   * @param tensorB the tensor to add.
   * @return a new tensor containing the elementwise sum.
   */
  public Tensor add(Tensor tensorB) {
    return add(this, tensorB);
  }

  /**
   * Adds two tensors elementwise.
   *
   * @param tensorA the first tensor.
   * @param tensorB the second tensor.
   * @return a new tensor containing the elementwise sum.
   */
  public static Tensor add(Tensor tensorA, Tensor tensorB) {
    return Ops.elementwise(tensorA, tensorB, (x, y) -> x + y, (x, y) -> 1.0, (x, y) -> 1.0);
  }

  /**
   * Subtracts another tensor from this tensor elementwise.
   *
   * @param tensorB the tensor to subtract.
   * @return a new tensor containing the elementwise difference.
   */
  public Tensor sub(Tensor tensorB) {
    return sub(this, tensorB);
  }

  /**
   * Subtracts one tensor from another elementwise.
   *
   * @param tensorA the base tensor.
   * @param tensorB the tensor to subtract.
   * @return a new tensor containing the elementwise difference.
   */
  public static Tensor sub(Tensor tensorA, Tensor tensorB) {
    return Ops.elementwise(tensorA, tensorB, (x, y) -> x - y, (x, y) -> 1.0, (x, y) -> -1.0);
  }

  /**
   * Computes the Hadamard product (elementwise multiplication) between this
   * tensor and another.
   *
   * @param tensorB the tensor to multiply by.
   * @return a new tensor containing the elementwise product.
   */
  public Tensor hadamard(Tensor tensorB) {
    return hadamard(this, tensorB);
  }

  /**
   * Computes the Hadamard product (elementwise multiplication) between two
   * tensors.
   *
   * @param tensorA the first tensor.
   * @param tensorB the second tensor.
   * @return a new tensor containing the elementwise product.
   */
  public static Tensor hadamard(Tensor tensorA, Tensor tensorB) {
    return Ops.elementwise(tensorA, tensorB, (x, y) -> x * y, (x, y) -> y, (x, y) -> x);
  }

  /**
   * Divides this tensor by another tensor elementwise.
   *
   * @param tensorB the denominator tensor.
   * @return a new tensor containing the elementwise quotient.
   */
  public Tensor div(Tensor tensorB) {
    return div(this, tensorB);
  }

  /**
   * Divides one tensor by another elementwise.
   *
   * @param tensorA the numerator tensor.
   * @param tensorB the denominator tensor.
   * @return a new tensor containing the elementwise quotient.
   */
  public static Tensor div(Tensor tensorA, Tensor tensorB) {
    Data.aboveValue(0, tensorB.core.core.data, "elementwise division");
    return Ops.elementwise(tensorA, tensorB, (x, y) -> x / y, (x, y) -> 1.0 / y, (x, y) -> -x / Math.pow(y, 2));
  }

  /**
   * Raises this tensor to the power of another tensor elementwise.
   *
   * @param tensorB the exponent tensor.
   * @return a new tensor containing the elementwise power results.
   */
  public Tensor pow(Tensor tensorB) {
    return pow(this, tensorB);
  }

  /**
   * Raises a base tensor to the power of an exponent tensor elementwise.
   *
   * @param tensorA the base tensor.
   * @param tensorB the exponent tensor.
   * @return a new tensor containing the elementwise power results.
   */
  public static Tensor pow(Tensor tensorA, Tensor tensorB) {
    return Ops.elementwise(tensorA, tensorB, (x, y) -> Math.pow(x, y), (x, y) -> y * Math.pow(x, y - 1), (x, y) -> Math.pow(x, y) * Math.log(x));
  }

  /**
   * Multiplies this tensor by another tensor using matrix multiplication.
   *
   * @param tensorB the multiplier tensor.
   * @return a new tensor containing the matrix product.
   */
  public Tensor matmul(Tensor tensorB) {
    return matmul(this, tensorB);
  }

  /**
   * Computes the matrix multiplication of two tensors.
   *
   * @param tensorA the first matrix tensor.
   * @param tensorB the second matrix tensor.
   * @return a new tensor containing the matrix product.
   */
  public static Tensor matmul(Tensor tensorA, Tensor tensorB) {
    Operation.verifyMatMul(tensorA.shape, tensorB.shape);

    int[] axesA = {tensorA.rank - 1};
    int[] axesB = {(tensorB.rank == 1) ? 0 : tensorB.rank - 2 };

    return contract(tensorA, tensorB, axesA, axesB);
  }

  // SHAPE OPERATIONS ---------------------------------

  /**
   * Broadcasts this tensor into a new target shape. Broadcasting a tensor is multiplying values of a tensor across
   * a new dimension to match the target shape layout. This is usually done implicitly and internally. This tensor must equal in rank
   * to the desired shape.
   *
   * @param shape the desired dimensions for the new tensor.
   * @return a new tensor matched to the target shape layout.
   */
  public Tensor broadcast(int... shape) {
    return broadcast(this, shape);
  }

  /**
   * Broadcasts a tensor into a new target shape. Broadcasting a tensor is multiplying values of a tensor across
   * a new dimension to match the target shape layout. This is usually done implicitly and internally.
   *
   * @param Tenor the tensor. Must equal in rank to the target shape.
   * @param shape the desired dimensions for the new tensor.
   * @return a new tensor matched to the target shape layout.
   */
  public static Tensor broadcast(Tensor tensor, int... shape) {
    return new Tensor(ShapeOps.broadcast(tensor.core, shape));
  }

  /**
   * Reshapes this tensor into a new target shape.
   *
   * @param shape the desired dimensions for the new tensor.
   * @return a new tensor matched to the target shape layout.
   */
  public Tensor reshape(int... shape) {
    return reshape(this, shape);
  }

  /**
   * Reshapes a specified tensor into a new target shape layout.
   *
   * @param tensor the tensor to reshape.
   * @param shape the desired dimensions for the new tensor.
   * @return a new tensor matched to the target shape layout.
   */
  public static Tensor reshape(Tensor tensor, int... shape) {
    return new Tensor(ShapeOps.reshape(tensor.core, shape));
  }

  /**
   * Permutes the axes order of this tensor according to the specified layout.
   *
   * @param newOrder the target ordering index map for the dimensions.
   * @return a new tensor with reordered axes.
   */
  public Tensor permute(int... newOrder) {
    return permute(this, newOrder);
  }

  /**
   * Permutes the axes order of a specified tensor according to the target layout.
   *
   * @param tensor the tensor to permute.
   * @param newOrder the target ordering index map for the dimensions.
   * @return a new tensor with reordered axes.
   */
  public static Tensor permute(Tensor tensor, int... newOrder) {
    Axes.verifyUniqueList("permutation", newOrder);
    CoreTensor out = ShapeOps.permute(tensor.core, newOrder);
    return new Tensor(out);
  }

  /**
   * Squeezes single-dimensional configurations (singletons) out of this tensor's dimensions at the specified axes.
   *
   * @param axes the location of singleton dimensions to remove.
   * @return a new tensor lacking singleton dimensions.
   */
  public Tensor squeeze(int... axes) {
    return squeeze(this, axes);
  }

  /**
   * Squeezes single-dimensional configurations (singletons) out of this tensor's dimensions at the specified axes.
   *
   * @param tensor the target tensor to squeeze.
   * @param axes the location of singleton dimensions to remove.
   * @return a new tensor lacking singleton dimensions.
   */
  public static Tensor squeeze(Tensor tensor, int... axes) {
    if (Reductions.prod(Tools.gather(tensor.shape, axes)) != 1) Message.crash("shaping", "cannot squeeze non-singleton dimension(s)", "selective squeezing", "tried to squeeze a non-singleton dimension (dimensions with size 1)");
    Axes.verifyAxes("squeezing", tensor.rank, axes);

    return new Tensor(ShapeOps.squeeze(tensor.core, axes));
  }

  /**
   * Squeezes single-dimensional configurations out of this tensor's dimensions.
   *
   * @return a new tensor lacking single dimensions.
   */
  public Tensor squeeze() {
    return squeeze(this);
  }

  /**
   * Squeezes single-dimensional configurations out of a specified tensor's
   * dimensions.
   *
   * @param tensor the target tensor to squeeze.
   * @return a new tensor lacking single dimensions.
   */
  public static Tensor squeeze(Tensor tensor) {
    return new Tensor(ShapeOps.squeeze(tensor.core));
  }

  /**
   * Unsqueezes a single dimension into a specified tensor's structural layout. The tensor will be expanded by iterating over the shape, and if an axis of unsqueezing is detected, will pad the shape with a singleton dimension at that location.
   *
   * @param axes the location of the new dimensions to expand.
   * @return a new tensor expanded by an isolated dimension.
   */
  public Tensor unsqueeze(int... axes) {
    return unsqueeze(this, axes);
  }

  /**
   * Unsqueezes a single dimension into a specified tensor's structural layout. The tensor will be expanded by iterating over the shape, and if an axis of unsqueezing is detected, will pad the shape with a singleton dimension at that location.
   *
   * @param tensor the target tensor to expand.
   * @param axes the location of the new dimensions to expand.
   * @return a new tensor expanded by an isolated dimension.
   */
  public static Tensor unsqueeze(Tensor tensor, int... axes) {
    Axes.verifyUniqueList("unsqueeze", axes);

    return new Tensor(ShapeOps.unsqueeze(tensor.core, axes));
  }

  /**
   * Extracts a specific dimensional slice from this tensor layout.
   *
   * @param axis the dimension line targeted for slicing.
   * @param index the exact index position inside the targeted axis slice.
   * @return a new tensor representing the extracted slice.
   */
  public Tensor slice(int axis, int index) {
    return slice(this, axis, index);
  }

  /**
   * Extracts a specific dimensional slice from a specified tensor structure.
   *
   * @param tensor the tensor to slice from.
   * @param axis the dimension line targeted for slicing.
   * @param index the exact index position inside the targeted axis slice.
   * @return a new tensor representing the extracted slice with removed dimension set to singletons.
   */
  public static Tensor slice(Tensor tensor, int axis, int index) {
    if (axis < 0 || axis >= tensor.rank) Message.crash("tensor indexing", "Axis out of bounds", "slicing", axis + " is out of bounds for a tensor of rank " + tensor.rank);
    if (index < 0 || index >= tensor.shape[axis]) Message.crash("tensor indexing", "Index out of bounds", "slicing", index + " is out of bounds for axis " + axis);
      
    return new Tensor(ShapeOps.slice(tensor.core, axis, index));
  }

  /**
   * Extracts specific dimensional slices from this tensor.
   *
   * @param axis the dimension line targeted for slicing.
   * @param indecies the exact positions inside the targeted axis slice.
   * @return a new tensor representing the extracted slice.
   */
  public Tensor gather(int axis, int... indecies) {
    return gather(this, axis, indecies);
  }

  /**
   * Extracts specific dimensional slices from a tensor.
   *
   * @param tensor the tensor to gather from.
   * @param axis the dimension line targeted for slicing.
   * @param indecies the exact positions inside the targeted axis slice.
   * @return a new tensor representing the extracted slice.
   */
  public static Tensor gather(Tensor tensor, int axis, int... indecies) {
    if (axis < 0 || axis >= tensor.rank) Message.crash("tensor indexing", "Axis out of bounds", "gather", axis + " is out of bounds for a tensor of rank " + tensor.rank);
    if (indecies.length == 0) Message.crash("tensor indexing", "Empty index", "gather", "tried to gather an empty index");

    Tensor[] tensors = new Tensor[indecies.length];
    for (int i = 0; i < tensors.length; i++) {
      if (indecies[i] < 0 || indecies[i] >= tensor.shape[axis]) Message.crash("tensor indexing", "Index out of bounds", "gather", indecies[i] + " is out of bounds for axis " + axis);
      tensors[i] = tensor.slice(axis, indecies[i]);
    }
    
    return concat(axis, tensors);
  }

  /**
   * Stacks an array of source tensors uniformly together along a targeted axis.
   *
   * @param axis the axis line sequence configuration where tensors stack together.
   * @param tensors the target group of tensors being joined.
   * @return a new stacked tensor representation.
   */
  public static Tensor stack(int axis, Tensor... tensors) {
    CoreTensor[] coreTensors = new CoreTensor[tensors.length];
    int[][] shapes = new int[tensors.length][];
    
    for (int i = 0; i < tensors.length; i++) {
      if (tensors[i].rank != tensors[0].rank) Message.crash("geometric", "ragged stacking", "stacking", "tried to stack tensors of varying ranks");
      if (tensors[i] == null) Message.crash("internal", "null tensor", "stacking", "tried to concatenate a null tensor");
      coreTensors[i] = tensors[i].core;
      shapes[i] = tensors[i].getShape();
    }
  
    Operation.verifyStack(shapes);
    if (axis < 0 || axis > tensors[0].rank - 1) Message.crash("shaping", "axis out of bounds", "stacking", "cannot stack tensors along axis " + axis + " because it is out of bounds");

    return new Tensor(ShapeOps.stack(axis, coreTensors));
  }

  /**
   * Concatenates an array of source tensors together sequentially along a
   * targeted axis.
   *
   * @param axis the operational axis sequence where arrays join up.
   * @param tensors the target group of tensors being concatenated.
   * @return a new unified continuous structural tensor.
   */
  public static Tensor concat(int axis, Tensor... tensors) {
    CoreTensor[] coreTensors = new CoreTensor[tensors.length];
    int[][] shapes = new int[tensors.length][];
    
    for (int i = 0; i < tensors.length; i++) {
      if (tensors[i].rank != tensors[0].rank) Message.crash("geometric", "ragged concatenation", "concatenation", "tried to concatenating tensors of varying ranks");
      if (tensors[i] == null) Message.crash("internal", "null tensor", "concatenation", "tried to concatenate a null tensor");
      coreTensors[i] = tensors[i].core;
      shapes[i] = tensors[i].getShape();
    }

    Operation.verifyConcat(axis, shapes);
    if (axis < 0 || axis > tensors[0].rank - 1) Message.crash("shaping", "axis out of bounds", "concatenation", "cannot concatenate tensors along axis " + axis + " because it is out of bounds");

    return new Tensor(ShapeOps.concat(axis, coreTensors));
  }

  /**
   * Flattens this tensor into a single-dimensional layout.
   *
   * @return a new flattened tensor.
   */
  public Tensor flatten() {
    return flatten(this);
  }

  /**
   * Flattens a tensor into a single-dimensional layout.
   *
   * @param tensor the tensor to flatten.
   * @return a new flattened tensor.
   */
  public static Tensor flatten(Tensor tensor) {
    return reshape(tensor, -1);
  }

  /**
   * Transposes the dimensional axes of this tensor.
   *
   * @return a new transposed tensor.
   */
  public Tensor transpose() {
    return transpose(this);
  }

  /**
   * Transposes the dimensional axes of a tensor. In this implementation, 
   *
   * @param tensor the tensor to transpose.
   * @return a new transposed tensor.
   */
  public static Tensor transpose(Tensor tensor) {
    if (tensor.rank < 2) Message.crash("mathematical", "transposing a vector or scalar", "tensor transpose", "the transpose of a vector or scalar is undefined");

    int[] transposed = Tools.arrange(tensor.rank);

    // swapping the last 2 axes
    int temp = transposed[tensor.rank - 1];
    transposed[tensor.rank - 1] = transposed[tensor.rank - 2];
    transposed[tensor.rank - 2] = temp;

    Tensor out = permute(tensor, transposed);

    return out;
  }
  
  // UNARY OPERATIONS ---------------------------------

  /**
   * Adds a scalar value to every element in this tensor.
   *
   * @param scalar the value to add.
   * @return a new shifted tensor.
   */
  public Tensor add(double scalar) {
    return add(this, scalar);
  }

  /**
   * Adds a scalar value to every element in a tensor.
   *
   * @param tensor the tensor to modify.
   * @param scalar the value to add.
   * @return a new shifted tensor.
   */
  public static Tensor add(Tensor tensor, double scalar) {
    return Ops.elementwise(tensor, (a) -> a + scalar, (a) -> 1.0);
  }

  /**
   * Subtracts a scalar value to every element in this tensor.
   *
   * @param scalar the value to subtract.
   * @return a new shifted tensor.
   */
  public Tensor sub(double scalar) {
    return sub(this, scalar);
  }

  /**
   * Subtracts a scalar value to every element in a tensor.
   *
   * @param tensor the tensor to modify.
   * @param scalar the value to subtract.
   * @return a new shifted tensor.
   */
  public static Tensor sub(Tensor tensor, double scalar) {
    return Ops.elementwise(tensor, (a) -> a - scalar, (a) -> 1.0);
  }

  /**
   * Multiplies every element in this tensor by a scalar value.
   *
   * @param scalar the scaling factor.
   * @return a new scaled tensor.
   */
  public Tensor mul(double scalar) {
    return mul(this, scalar);
  }

  /**
   * Multiplies every element in a tensor by a scalar value.
   *
   * @param tensor the tensor to scale.
   * @param scalar the scaling factor.
   * @return a new scaled tensor.
   */
  public static Tensor mul(Tensor tensor, double scalar) {
    return Ops.elementwise(tensor, (a) -> a * scalar, (a) -> scalar);
  }

  /**
   * Divides every element in this tensor by a scalar value.
   *
   * @param scalar the scaling factor.
   * @return a new scaled tensor.
   */
  public Tensor div(double scalar) {
    return div(this, scalar);
  }

  /**
   * Divides every element in a tensor by a scalar value.
   *
   * @param tensor the tensor to scale.
   * @param scalar the dividing factor.
   * @return a new scaled tensor.
   */
  public static Tensor div(Tensor tensor, double scalar) {
    if (scalar == 0.0) Message.crash("mathematical", "division by zero", "scalar division", "tried to divide tensor data by zero");
    return Ops.elementwise(tensor, (a) -> a / scalar, (a) -> scalar);
  }

  /**
   * Raises every element in this tensor to a scalar power.
   *
   * @param power the exponent value.
   * @return a new exponentiated tensor.
   */
  public Tensor pow(double power) {
    return pow(this, power);
  }

  /**
   * Raises every element in a tensor to a scalar power.
   *
   * @param tensor the base tensor.
   * @param scalar the exponent value.
   * @return a new exponentiated tensor.
   */
  public static Tensor pow(Tensor tensor, double power) {
    return Ops.elementwise(tensor, (a) -> Math.pow(a, power), (a) -> power * Math.pow(a, power - 1.0));
  }

  /**
   * Exponentiates every element in this tensor to a scalar power (scalar^element).
   *
   * @param base the base value.
   * @return a new exponentiated tensor.
   */
  public Tensor exp(double base) {
    return exp(this, base);
  }

  /**
   * Exponentiates every element in this tensor to a scalar power (scalar^element).
   *
   * @param tensor the exponent tensor.
   * @param base the base value.
   * @return a new exponentiated tensor.
   */
  public static Tensor exp(Tensor tensor, double base) {
    if (base < 0) Message.crash("tensor argument", "Illegal parameter", "exponentiation" , "parameter must be at least 0 but got " + base);
    return Ops.elementwise(tensor, (a) -> Math.pow(base, a), (a) -> Math.pow(base, a) * Math.log(base));
  }

  /**
   * Exponentiates every element in this tensor.
   *
   * @return a new exponentiated tensor.
   */
  public Tensor exp() {
    return exp(this);
  }

  /**
   * Exponentiates every element in this tensor.
   *
   * @param tensor the base tensor.
   * @return a new exponentiated tensor.
   */
  public static Tensor exp(Tensor tensor) {
    return Ops.elementwise(tensor, (a) -> Math.exp(a), (a) -> Math.exp(a));
  }

  /**
   * Take the logarithm of every element in this tensor using a specified base.
   *
   * @return a new log tensor.
   */
  public Tensor log(double base) {
    return log(this, base);
  }

  /**
   * Exponentiates every element in this tensor using a specified base.
   *
   * @param tensor the base tensor.
   * @param base the base of the logarithm.
   * @return a new exponentiated tensor.
   */
  public static Tensor log(Tensor tensor, double base) {
    Data.aboveValue(0, tensor.core.core.data, "log");
    Argument.aboveValue(0, base, "log");
    return Ops.elementwise(tensor, (a) -> Math.log(a)/Math.log(base), (a) -> 1.0 / (a * Math.log(base)));
  }

  /**
   * Take the logarithm of every element in this tensor using a specified base.
   *
   * @return a new log tensor.
   */
  public Tensor ln() {
    return ln(this);
  }

  /**
   * Exponentiates every element in this tensor using a specified base.
   *
   * @param tensor the base tensor.
   * @param base the base of the logarithm.
   * @return a new exponentiated tensor.
   */
  public static Tensor ln(Tensor tensor) {
    Data.aboveValue(0, tensor.core.core.data, "ln (natural log)");
    return Ops.elementwise(tensor, (a) -> Math.log(a), (a) -> 1.0 / a);
  }

  /**
   * Takes the square root of every element in this tensor.
   *
   * @param tensor the base tensor.
   * @param root the root value.
   * @return a new exponentiated tensor.
   */
  public Tensor root() {
    return root(this);
  }

  /**
   * Takes the square root of every element in this tensor.
   *
   * @param tensor the base tensor.
   * @param root the root value.
   * @return a new exponentiated tensor.
   */
  public static Tensor root(Tensor tensor) {
    Data.aboveValue(0, tensor.core.core.data, "square root");
    return Ops.elementwise(tensor, (a) -> Math.pow(a, 0.5), (a) -> 0.5 * Math.pow(a, -1.5));
  }

  /**
   * Takes the root of every element in this tensor.
   *
   * @param tensor the base tensor.
   * @param root the root value.
   * @return a new exponentiated tensor.
   */
  public Tensor root(double root) {
    return root(this, root);
  }

  /**
   * Takes the root of every element in this tensor.
   *
   * @param tensor the base tensor.
   * @param root the root value.
   * @return a new exponentiated tensor.
   */
  public static Tensor root(Tensor tensor, double root) {
    Data.aboveValue(0, tensor.core.core.data, "root");
    return Ops.elementwise(tensor, (a) -> Math.pow(a, 1.0 / root), (a) -> 1.0 / root * Math.pow(a, 1.0 / root - 1.0));
  }
}
