package com.aufy.jnet.core.tensor.raw.main;

import com.aufy.jnet.core.tensor.core.backend.compute.Engine;
import com.aufy.jnet.core.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;

public class RawShapeOps {

  public static RawTensor permute(RawTensor tensor, int[] newOrder) {
    int[] newShape = Shaping.reorder(tensor.shape, newOrder);
    int[] newStrides = Shaping.reorder(tensor.strides, newOrder);

    double[] newData = Engine.makeContiguous(tensor.dump(), newShape, newStrides);
    
    return new RawTensor(newData, newShape);
  }
  
  public static RawTensor broadcast(RawTensor tensor, int[] targetShape) {
    if (java.util.Arrays.equals(tensor.shape, targetShape)) { // dont waste compute
      return tensor;
    }
    
    double[] broadcastedData = Engine.broadcast(tensor.dump(), tensor.strides, tensor.shape, targetShape);
    return new RawTensor(broadcastedData, targetShape);
  }
  
  public static RawTensor squeeze(RawTensor tensor) {
    return new RawTensor(tensor.dump(), Shaping.squeeze(tensor.shape));
  }

  public static RawTensor squeeze(RawTensor tensor, int[] axes) {
    return new RawTensor(tensor.dump(), Shaping.removeAxes(tensor.shape, axes));
  }
  
  public static RawTensor unsqueeze(RawTensor tensor, int[] axes) {
    return new RawTensor(tensor.dump(), Shaping.unsqueeze(tensor.shape, axes));
  }
  
  public static RawTensor reshape(RawTensor tensor, int[] shape) {    
    int[] newShape = Shaping.inferShape(tensor.shape, shape);
    
    return new RawTensor(tensor.dump(), newShape);
  }
  
  public static RawTensor concat(int axis, RawTensor[] tensors) {
    int[][] shapes = new int[tensors.length][];
    double[][] dataList = new double[tensors.length][];
    int[] resShape = tensors[0].shape.clone();
    int totalAxisDim = 0;

    for (int i = 0; i < tensors.length; i++) {
      shapes[i] = tensors[i].shape;
      dataList[i] = tensors[i].dump();
      totalAxisDim += shapes[i][axis];
    }
    
    resShape[axis] = totalAxisDim;

    double[] resultData = Engine.concat(axis, shapes, dataList, resShape);

    return new RawTensor(resultData, resShape);
  }

  public static RawTensor stack(int axis, RawTensor[] tensors) {
    int[] resShape = Shaping.unsqueeze(tensors[0].shape, new int[]{axis});
    double[][] dataList = new double[tensors.length][];
    int[][] shapes = new int[tensors.length][];
    
    resShape[axis] = tensors.length;
    for (int i = 0; i < tensors.length; i++) dataList[i] = tensors[i].dump();
    for (int i = 0; i < tensors.length; i++) shapes[i] = tensors[i].shape;

    double[] resultData = Engine.stack(axis, dataList, shapes, resShape);

    return new RawTensor(resultData, resShape);
  }

  public static RawTensor slice(RawTensor tensor, int axis, int index) {
    int[] newShape = Shaping.reduceAxes(tensor.shape, new int[]{axis});
    double[] resultData = Engine.slice(tensor.dump(), tensor.shape, tensor.strides, axis, index);

    return new RawTensor(resultData, newShape);
  }

}
