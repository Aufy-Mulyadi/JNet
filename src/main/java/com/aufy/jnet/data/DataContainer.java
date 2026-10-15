package com.aufy.jnet.data;

import com.aufy.jnet.jtensor.Tensor;

/**
 * Holds features and labels for a DataContainer object, both of which are of type Tensor. Contains nothing more, exists for convenience.
 */
public class DataContainer {
  private final Tensor features;
  private final Tensor labels;

  public DataContainer(Tensor features, Tensor labels) {
    this.features = features;
    this.labels = labels;
  }

  public DataContainer select(int... indices) {
    return new DataContainer(features.gather(0, indices), labels.gather(0, indices));
  }

  public int size() {
    return features.getShape(0);
  }

  public Tensor features(int index) {
    return features.slice(0, index);
  }

  public Tensor labels(int index) {
    return labels.slice(0, index);
  }

  public Tensor features() {
    return features;
  }

  public Tensor labels() {
    return labels;
  }
}
