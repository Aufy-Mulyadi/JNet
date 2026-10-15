package com.aufy.jnet.core.tensor.graph.main;

import java.util.Arrays;
import java.util.List;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.arrayops.Tools;
import com.aufy.jnet.core.backend.arrayops.Uniform;
import com.aufy.jnet.core.backend.interfaces.atomic.BinaryOp;
import com.aufy.jnet.core.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.core.tensor.core.implementations.CoreTensor;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;
import com.aufy.jnet.core.tensor.raw.main.RawBinaryOps;
import com.aufy.jnet.core.tensor.raw.main.RawShapeOps;
import com.aufy.jnet.core.tensor.raw.main.RawSubtensorOps;

public class BinaryOps {
  
  public static CoreTensor elementwise(CoreTensor tensorA, CoreTensor tensorB, BinaryOp operation, BinaryOp dA, BinaryOp dB) {
    CoreTensor out = new CoreTensor(RawBinaryOps.elementwise(tensorA.core, tensorB.core, operation));
    
    out.requiresGrad = tensorA.requiresGrad || tensorB.requiresGrad;
    
    if (out.requiresGrad) {
      out.parents = List.of(tensorA, tensorB);
      
      out.derivative = (grad) -> {
        if (tensorA.requiresGrad) {
          RawTensor dACore = RawBinaryOps.elementwise(tensorA.core, tensorB.core, dA); // dy/dx
          double[] gradA = BinaryOp.apply(grad.data, dACore.data, (x, y) -> x * y); // grad * dy/dx = grad/dx

          if (!Arrays.equals(tensorA.shape, tensorB.shape)) {
            RawTensor reducedData = RawSubtensorOps.reduce(new RawTensor(gradA, dACore.shape), Reductions::sum, (Shaping.broadcastedAxes(tensorA.shape, tensorB.shape)));
            tensorA.accumulate(reducedData);
          } else {
            tensorA.accumulate(gradA);
          }
          
        }
        
        if (tensorB.requiresGrad) {
          RawTensor dBCore = RawBinaryOps.elementwise(tensorA.core, tensorB.core, dB);
          double[] gradB = BinaryOp.apply(grad.data, dBCore.data, (x, y) -> x * y);

          if (!Arrays.equals(tensorA.shape, tensorB.shape)) {
            RawTensor reducedData = RawSubtensorOps.reduce(new RawTensor(gradB, dBCore.shape), Reductions::sum, (Shaping.broadcastedAxes(tensorB.shape, tensorA.shape)));
            tensorB.accumulate(reducedData);
          } else {
            tensorB.accumulate(gradB);
          }
        }
      };
    }
    
    return out;
  }
  
  public static CoreTensor contract(CoreTensor tensorA, CoreTensor tensorB, int[] axesA, int[] axesB) {
    CoreTensor out = new CoreTensor(RawBinaryOps.contract(tensorA.core, tensorB.core, axesA, axesB));
    out.requiresGrad = tensorA.requiresGrad || tensorB.requiresGrad;
    
    if (out.requiresGrad) {
      out.parents = List.of(tensorA, tensorB);
      
      int[] freeA = Shaping.removeAxes(Tools.arrange(tensorA.rank), axesA);
      int[] freeB = Shaping.removeAxes(Tools.arrange(tensorB.rank), axesB);

      out.derivative = (grad) -> {
        if (tensorA.requiresGrad) {
          RawTensor gradACore = RawBinaryOps.contract(
            grad, 
            tensorB.core,
            Uniform.add(Tools.arrange(freeB.length), freeA.length),
            freeB
          );

          int[] permuteOrder = new int[tensorA.rank];

          for (int i = 0; i < tensorA.rank; i++) {
            permuteOrder[Tools.concat(freeA, axesA)[i]] = i;
          }

          tensorA.accumulate(RawShapeOps.permute(gradACore, permuteOrder));
        }
        if (tensorB.requiresGrad) {
          RawTensor gradBCore = RawBinaryOps.contract(
            tensorA.core, 
            grad, 
            freeA,
            Tools.arrange(freeA.length)
          );

          int[] permuteOrder = new int[tensorB.rank];

          for (int i = 0; i < tensorB.rank; i++) {
            permuteOrder[Tools.concat(axesB, freeB)[i]] = i;
          }

          tensorB.accumulate(RawShapeOps.permute(gradBCore, permuteOrder));
        }
      };
    }
    
    return out;
  }
}
