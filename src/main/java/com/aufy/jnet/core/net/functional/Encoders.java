package com.aufy.jnet.core.net.functional;

public class Encoders {

  public static double[] onehot(double value, double[] categories) {
    double[] encoded = new double[categories.length];
    for (int i = 0; i < categories.length; i++) {
      if (Math.abs(categories[i] - value) < 1e-9) encoded[i] = 1.0;
    }
    return encoded;
  }
}
