package com.aufy.jnet.data;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

import com.aufy.jnet.core.backend.arrayops.Tools;
import com.aufy.jnet.jtensor.Tensor;

public class FileLoader {
  /**
   * Reads a CSV file into a Container object with Tensor features and labels. This method will not remove nor ensure file
   * compatibility.
   * 
   * @param filePath the path to the CSV file.
   * @param labelColumn the column indices of the labels.
   * @param labelShape the shape of the labels.
   * @param featureColumn the column indices of the features.
   * @param featureShape the shape of the features.
   * @return Container object with parameters "features" and "labels."
   */
  public static DataContainer loadCSV(String filePath, int[] labelColumn, int[] labelShape, int[] featureColumn, int[] featureShape) {
    ArrayList<Tensor> features = new ArrayList<>();
    ArrayList<Tensor> labels = new ArrayList<>();

    try (BufferedReader br = Files.newBufferedReader(Path.of(filePath))) {

      String line;

      while ((line = br.readLine()) != null) {
        String[] parts = line.split(",");
        double[] featureRowData = new double[featureColumn.length];
        double[] labelRowData  = new double[labelColumn.length];
        
        int sourceInex = 0;
        int featureIndex = 0;
        int labelIndex = 0;
        while (sourceInex < parts.length) {
          if (Tools.contains(featureColumn, sourceInex)) {
            featureRowData[featureIndex] = Double.parseDouble(parts[sourceInex]);
            featureIndex++;
            sourceInex++;
          }
          
          if (Tools.contains(labelColumn, sourceInex)) {
            labelRowData[labelIndex] = Double.parseDouble(parts[sourceInex]);
            sourceInex++;
            labelIndex++;
          } 
        }

        features.add(new Tensor(featureRowData, Tools.concat(new int[]{1}, featureShape)));
        labels.add(new Tensor(labelRowData, Tools.concat(new int[]{1}, labelShape)));
      }
    } catch (Exception e) {
      System.exit(0);
    }

    Tensor featureTensor = Tensor.concat(0, features.toArray(Tensor[]::new));
    Tensor labelTensor  = Tensor.concat(0, labels.toArray(Tensor[]::new));

    return new DataContainer(featureTensor, labelTensor);
  }

  /**
   * Reads a CSV file into a Container object with Tensor features and labels. This method will not remove nor ensure file
   * compatibility.
   * 
   * @param filePath the path to the CSV file.
   * @param labelColumn the column indices of the labels.
   * @param labelShape the shape of the labels.
   * @param featureShape the shape of the features.
   * @return Container object with parameters "features" and "labels."
   */
  public static DataContainer loadCSV(String filePath, int[] labelColumn, int[] labelShape, int[] featureShape) {
    ArrayList<Tensor> features = new ArrayList<>();
    ArrayList<Tensor> labels = new ArrayList<>();

    try (BufferedReader br = Files.newBufferedReader(Path.of(filePath))) {

      String line;

      while ((line = br.readLine()) != null) {
        String[] parts = line.split(",");
        double[] featureRowData = new double[parts.length - labelColumn.length];
        double[] labelRowData  = new double[labelColumn.length];
        
        int sourceInex = 0;
        int featureIndex = 0;
        int labelIndex = 0;
        while (sourceInex < parts.length) {
          
          if (Tools.contains(labelColumn, sourceInex)) {
            labelRowData[labelIndex] = Double.parseDouble(parts[sourceInex]);
            sourceInex++;
            labelIndex++;
          } else {
            featureRowData[featureIndex] = Double.parseDouble(parts[sourceInex]);
            featureIndex++;
            sourceInex++;
          }
        }

        features.add(new Tensor(featureRowData, Tools.concat(new int[]{1}, featureShape)));
        labels.add(new Tensor(labelRowData, Tools.concat(new int[]{1}, labelShape)));
      }
    } catch (Exception e) {
      System.exit(0);
    }

    Tensor featureTensor = Tensor.concat(0, features.toArray(Tensor[]::new));
    Tensor labelTensor  = Tensor.concat(0, labels.toArray(Tensor[]::new));

    return new DataContainer(featureTensor, labelTensor);
  }

}
