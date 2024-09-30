//package org.a1.q2;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class KMeansClusteringMR {

    public static class KMeansMapper extends Mapper<LongWritable, Text, Text, Text> {
        private List<double[]> centroids = new ArrayList<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            Configuration conf = context.getConfiguration();
            int k = conf.getInt("k", 2); // Default to 2 clusters if not specified
            for (int i = 0; i < k; i++) {
                String[] centroid = conf.get("centroid." + i).split(",");
                centroids.add(new double[]{Double.parseDouble(centroid[0]), Double.parseDouble(centroid[1])});
            }
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] point = value.toString().split(",");
            double x = Double.parseDouble(point[0]);
            double y = Double.parseDouble(point[1]);

            int closestCentroidIndex = findClosestCentroid(x, y);
            context.write(new Text(String.valueOf(closestCentroidIndex)), new Text(x + "," + y));
        }

        private int findClosestCentroid(double x, double y) {
            int closestIndex = 0;
            double minDistance = Double.MAX_VALUE;

            for (int i = 0; i < centroids.size(); i++) {
                double[] centroid = centroids.get(i);
                double distance = Math.sqrt(Math.pow(x - centroid[0], 2) + Math.pow(y - centroid[1], 2));
                if (distance < minDistance) {
                    minDistance = distance;
                    closestIndex = i;
                }
            }

            return closestIndex;
        }
    }

    public static class KMeansReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            double sumX = 0, sumY = 0;
            int count = 0;

            for (Text value : values) {
                String[] point = value.toString().split(",");
                sumX += Double.parseDouble(point[0]);
                sumY += Double.parseDouble(point[1]);
                count++;
            }

            double newX = sumX / count;
            double newY = sumY / count;

            context.write(key, new Text(newX + "," + newY));
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            System.err.println("Usage: KMeansClusteringMR <input path> <output path> <number of clusters> <max iterations>");
            System.exit(-1);
        }

        String inputPath = args[0];
        String outputPath = args[1];
        int k = Integer.parseInt(args[2]);
        int maxIterations = Integer.parseInt(args[3]);

        Configuration conf = new Configuration();
        conf.setInt("k", k);

        // Initialize random centroids (you may want to read these from a file for reproducibility)
        for (int i = 0; i < k; i++) {
            double randomX = Math.random() * 100; // Adjust range as needed
            double randomY = Math.random() * 100; // Adjust range as needed
            conf.set("centroid." + i, randomX + "," + randomY);
        }

        long startTime = System.currentTimeMillis();
        boolean hasConverged = false;
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            Job job = Job.getInstance(conf, "K-means Clustering Iteration " + iteration);
            job.setJarByClass(KMeansClusteringMR.class);
            job.setMapperClass(KMeansMapper.class);
            job.setReducerClass(KMeansReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);

            FileInputFormat.addInputPath(job, new Path(inputPath)); // Use datapoint.txt as the input
            FileOutputFormat.setOutputPath(job, new Path(outputPath + "_" + iteration));

            job.waitForCompletion(true);

            // Update centroids for the next iteration
            hasConverged = updateCentroids(conf, outputPath + "_" + iteration + "/part-r-00000");

            // Check for convergence or time limit
            long elapsedTime = System.currentTimeMillis() - startTime;
            if (hasConverged || elapsedTime > 120000) { // 120 seconds time limit
                break;
            }
        }
        System.out.println("K-means clustering completed in " + ((System.currentTimeMillis() - startTime) / 1000.0) + " seconds.");
    }
    private static boolean updateCentroids(Configuration conf, String newCentroidsPath) {
        // Read the new centroids from the output file and update the configuration
        // This is a simplified version. In practice, you'd need to use Hadoop's file system API to read the file.
        // For demonstration purposes, we'll just print a message and check for convergence.
        System.out.println("Updating centroids from: " + newCentroidsPath);

        // Check for convergence (a simple implementation)
        boolean hasConverged = true;
        int k = conf.getInt("k", 2);
        for (int i = 0; i < k; i++) {
            String[] oldCentroid = conf.get("centroid." + i).split(",");
            String[] newCentroid = new String[2];
            // Read the new centroid and compare it with the old one
            // If any centroid has changed, set hasConverged to false
            newCentroid[0] = "50.0"; // replace with actual new value
            newCentroid[1] = "50.0"; // replace with actual new value
            if (Math.abs(Double.parseDouble(oldCentroid[0]) - Double.parseDouble(newCentroid[0])) > 0.01 ||
                    Math.abs(Double.parseDouble(oldCentroid[1]) - Double.parseDouble(newCentroid[1])) > 0.01) {
                hasConverged = false;
                break;
            }
            conf.set("centroid." + i, newCentroid[0] + "," + newCentroid[1]);
        }

        return hasConverged;
    }
}