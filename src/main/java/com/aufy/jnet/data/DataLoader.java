package com.aufy.jnet.data;

import java.util.Arrays;
import java.util.Iterator;

import com.aufy.jnet.core.backend.arrayops.Tools;

/**
 * Loads Data from a DataContainer object into batches following iteration rules with iterator capabilities. Only transforms 
 * the DataContainer object and nothing else.
 */
public class DataLoader implements Iterable<DataContainer> {
  private final DataContainer dataset;
  private final int batchSize;
  private final boolean shuffle;

  public DataLoader(DataContainer dataset, int batchSize, boolean shuffle) {
    this.dataset = dataset;
    this.batchSize = batchSize;
    this.shuffle = shuffle;
  }

  @Override
  public Iterator<DataContainer> iterator() {
    int[] indices = (shuffle) ? Tools.shuffle(Tools.arrange(this.dataset.size())) : Tools.arrange(this.dataset.size());

    return new Iterator<DataContainer>() {
      private int position = 0;

      @Override
      public boolean hasNext() {
        return position < indices.length;
      }

      @Override
      public DataContainer next() {
        int end = Math.min(position + batchSize, indices.length);

        int[] batch = Arrays.copyOfRange(indices, position, end);
        position = end;

        return dataset.select(batch);
      }
    };
  }

}
