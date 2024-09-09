import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MockDataGenerator {
	Random random = new Random();
	int numDataPoints = 1000;
	int numOutliers = 50;
	double normalMean = 50;
	double normalDeviation = 10;
	double outlierMean = 100;
	double outlierDeviation = 20;

	MockDataGenerator(int numDataPoints, int numOutliers, double normalMean, double normalDeviation, double outlierMean,
			double outlierDeviation) {
		this.numDataPoints = numDataPoints;
		this.numOutliers = numOutliers;
		this.normalMean = normalMean;
		this.normalDeviation = normalDeviation;
		this.outlierMean = outlierMean;
		this.outlierDeviation = outlierDeviation;
	}

	void generate(String inputDataFile) {
		int totalOutliers = 0;
		try (FileWriter writer = new FileWriter(inputDataFile)) {
			writer.write("value,outlier\n");
			;
			for (int i = 0; i < numDataPoints; i++) {
				double alpha = random.nextDouble() * numDataPoints;
				boolean isOutlier = alpha < numOutliers;
				double v = isOutlier ? outlierMean + random.nextGaussian() * outlierDeviation
						: normalMean + random.nextGaussian() * normalDeviation;
				writer.write(String.format("%.2f,%s\n", v, isOutlier ? "true" : "false"));
				if (isOutlier)
					totalOutliers++;
			}

			// Apparently you can shuffle the data if you save the data to an ArrayList of
			// doubles
			// Collections.shuffle(data, random);

		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("# of outliers: " + String.valueOf(totalOutliers));
	}

	public static void main(String[] args) {
		System.out.println("Hello world!");
		MockDataGenerator g = new MockDataGenerator(500, 75, 50, 25, 150, 75);
		g.generate("data_with_outliers.csv");
		System.out.println("Goodbye!");
	}
}
