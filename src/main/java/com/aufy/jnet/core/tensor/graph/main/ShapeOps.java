package com.aufy.jnet.core.tensor.graph.main;

import java.util.List;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.tensor.core.backend.compute.Engine;
import com.aufy.jnet.core.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.core.tensor.core.implementations.CoreTensor;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;
import com.aufy.jnet.core.tensor.raw.main.RawShapeOps;
import com.aufy.jnet.core.tensor.raw.main.RawSubtensorOps;

public class ShapeOps {
  
  public static CoreTensor permute(CoreTensor tensor, int[] axes) {
    CoreTensor out = new CoreTensor(RawShapeOps.permute(tensor.core, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      int[] inverse = new int[axes.length];

      for (int i = 0; i < axes.length; i++) {
        inverse[axes[i]] = i;
      }

      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        tensor.accumulate(RawShapeOps.permute(grad, inverse));
      };
    }

    return out;
  }

  public static CoreTensor broadcast(CoreTensor tensor, int[] targetShape) {
    if (java.util.Arrays.equals(tensor.shape, targetShape)) { // dont waste compute
      return tensor;
    }
    
    CoreTensor out = new CoreTensor(RawShapeOps.broadcast(tensor.core, targetShape));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        tensor.accumulate(RawSubtensorOps.reduce(grad, 
        Reductions::sum, 
        // Tools.findIndices(tensor.shape, 1) // indeces that were broadcasted
        Shaping.broadcastedAxes(tensor.shape, targetShape) // indeces that were broadcasted
        ));
      };
    }

    return out;
  }

  public static CoreTensor reshape(CoreTensor tensor, int[] shape) {
    int[] newShape = Shaping.inferShape(tensor.shape, shape);
    CoreTensor out = new CoreTensor(RawShapeOps.reshape(tensor.core, newShape));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        tensor.accumulate(RawShapeOps.reshape(grad, tensor.shape));
      };
    }

    return out;
  }

  public static CoreTensor squeeze(CoreTensor tensor) {
    CoreTensor out = new CoreTensor(RawShapeOps.squeeze(tensor.core));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);
      
      out.derivative = (grad) -> {
        // RawTensor reshaped = RawShapeOps.reshape(grad, tensor.shape);
        tensor.accumulate(grad.data);
      };
    }
    
    return out;
  }

  public static CoreTensor squeeze(CoreTensor tensor, int[] axes) {
    CoreTensor out = new CoreTensor(RawShapeOps.squeeze(tensor.core, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);
      
      out.derivative = (grad) -> {
        // RawTensor reshaped = RawShapeOps.reshape(grad, tensor.shape);
        tensor.accumulate(grad.data);
      };
    }
    
    return out;
  }
  
  public static CoreTensor unsqueeze(CoreTensor tensor, int[] axes) {
    CoreTensor out = new CoreTensor(RawShapeOps.unsqueeze(tensor.core, axes));
    out.requiresGrad = tensor.requiresGrad;
    
    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        // RawTensor reshaped = RawShapeOps.reshape(grad, tensor.shape);
        tensor.accumulate(grad.data);
      };
    }

    return out;
  }
  
  public static CoreTensor concat(int axis, CoreTensor[] tensors) {
    RawTensor[] tensorsData = new RawTensor[tensors.length];
    boolean requiresGrad = false;

    for (int i = 0; i < tensors.length; i++) {
      tensorsData[i] = tensors[i].core;
      if (tensors[i].requiresGrad) requiresGrad = true;
    }

    CoreTensor out = new CoreTensor(RawShapeOps.concat(axis, tensorsData));
    out.requiresGrad = requiresGrad;

    if (requiresGrad) {
      out.parents = List.of(tensors);
      
      out.derivative = (grad) -> {
        int offset = 0;
        for (CoreTensor parent : tensors) {
          int dimSize = parent.shape[axis];

          double[] partialGrad = Engine.rangedSlice(grad.dump(), grad.shape, grad.strides, axis, offset, offset + dimSize);

          parent.accumulate(partialGrad);
          offset += dimSize;
        }
      };
    }

    return out;
  }

  public static CoreTensor stack(int axis, CoreTensor[] tensors) {
    RawTensor[] tensorsData = new RawTensor[tensors.length];
    boolean requiresGrad = false;

    for (int i = 0; i < tensors.length; i++) {
      tensorsData[i] = tensors[i].core;
      if (tensors[i].requiresGrad) requiresGrad = true;
    }

    CoreTensor out = new CoreTensor(RawShapeOps.stack(axis, tensorsData));
    out.requiresGrad = requiresGrad;

    if (requiresGrad) {
      out.parents = List.of(tensors);
      
      out.derivative = (grad) -> {
        for (int i = 0; i < tensors.length; i++) {
          // Since stack added a dimension, a single-index slice returns the original shape
          double[] partialGrad = Engine.slice(grad.dump(), grad.shape, grad.strides, axis, i);
          
          tensors[i].accumulate(partialGrad);
        }
      };
    }

    return out;
  }
  
  public static CoreTensor slice(CoreTensor tensor, int axis, int index) {
    CoreTensor out = new CoreTensor(RawShapeOps.slice(tensor.core, axis, index));
    out.requiresGrad = tensor.requiresGrad;
  
    if (out.requiresGrad) {
      out.parents = List.of(tensor);
      
      out.derivative = (grad) -> {
        // Gradient is smaller than input. Place it in a zero-filled array of original shape.
        double[] expandedGradData = Engine.insert(grad.dump(), tensor.core.shape, tensor.core.strides, axis, index);
        
        tensor.accumulate(expandedGradData);
      };
    }
  
    return out;
  }
}
