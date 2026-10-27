import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;

public class DataProcessingorIris {
  public static void main(String[] args) {
    try (BufferedReader br = Files.newBufferedReader(Path.of("/Users/72212/Desktop/Homework/Coding/java/JNet/src/test/resources/irisData.csv"))) {
      String line;
      int row = 0;
      int errors = 0;
      double item;

      HashSet<String> seen = new HashSet<>();

      while ((line = br.readLine()) != null) {
        row++;
        String[] parts = line.trim().split(",");

        if (seen.contains(line.trim().toLowerCase())) {
          errors++;
          // System.out.println(errors + " Duplicate row " + row);
        } else {
          seen.add(line.trim().toLowerCase());
        }
        
        String label = parts[4];
        if (!label.trim().equalsIgnoreCase("Iris-virginica") 
         && !label.trim().equalsIgnoreCase("Iris-versicolor") 
         && !label.trim().equalsIgnoreCase("Iris-setosa")) {
          errors++;
          // System.out.println(errors + " Label error '" + label + "' - column 4 row " + row);
        }

        for (int i = 0; i < 4; i++) {
          String part = parts[i];
          try {
            item = Double.valueOf(part);
            if (item<0) {
              errors++;
              System.out.println(errors + " Impossible value '" + part + "' - column " + (i + 1) + " row " + row);
            }
          } catch (Exception e) {
            errors++;
            // System.out.println(errors + " Formatting error '" + part + "' - column " + (i + 1) + " row " + row);
          } 
        }
      }

    } catch (Exception e) {
      System.exit(0);
    }
  }
}
