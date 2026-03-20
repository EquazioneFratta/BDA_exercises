import java.io.IOException;

import javax.naming.Context;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class HadoopWordPairs extends Configured implements Tool {

	public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {
		
		private final static IntWritable one = new IntWritable(1);
		private Text pair = new Text();
		private Text lastWord = new Text();

		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

			value.toString().toLowerCase();
			/*
			 * String[] splitLine = value.toString().split()
			 * Starting from this line, write me the regex to split words as follow:
			 * words with only lower case letters, including the 2 dashes "-" "_"
			 * numbers consisting of sequences of digits 0-9 including "."",", plus at most
			 * one leading dash "-"
			 */
			String[] splitLine = value.toString()
					.split("[^a-z0-9.,_-]+|(?<=[a-z_-])(?=[0-9.-])|(?<=[0-9.,])(?=[a-z_])");
			for (String w : splitLine) {
				if (lastWord.getLength() > 0) {
					pair.set(lastWord + ":" + w);
					context.write(pair, one);
				}
				lastWord.set(w);
			}
		}
	}

	public static class Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {

		@Override
		public void reduce(Text key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;

			for (IntWritable value : values)
				sum += value.get();

			context.write(key, new IntWritable(sum));
		}
	}

	@Override
	public int run(String[] args) throws Exception {
		Job job = Job.getInstance(new Configuration(), "HadoopWordPairs");
		job.setJarByClass(HadoopWordPairs.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		job.setMapperClass(Map.class);
		job.setCombinerClass(Reduce.class);
		job.setReducerClass(Reduce.class);

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		FileInputFormat.setInputPaths(job, args[0]);
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		job.waitForCompletion(true);
		return 0;
	}

	public static void main(String[] args) throws Exception {
		int ret = ToolRunner.run(new Configuration(), new HadoopWordPairs(), args);
		System.exit(ret);
	}
}
