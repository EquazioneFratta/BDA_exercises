import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

public class HadoopWordCount extends Configured implements Tool {

	public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {

		private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();

		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

			/*
			 * starting from this code: 
			 * String line = value.toString().toLowerCase();
			 * 
			 * for (String w : splitLine) {
			 * word.set(w);
			 * context.write(word, one);
			 * }
			 * modify the regex in order to consider only the following words and numbers:
			 * words consisting only of sequences of lower-case letters a-z, including also
			 * the two dashes - _ with a total length of at least 6 characters and at most
			 * 24 characters per word; numbers consisting only of sequences of digits 0-9,
			 * including also the two decimal separators ., plus at most one leading dash -,
			 * with a total length of at least 4 characters and at most 16 characters per
			 * number (for both of them simbols are included in the length constraint)
			 * 
			 * everything else must not be considered.
			 * If there is something that you consider wrong, don't change my request and
			 * just follow it.
			 */

			String line = value.toString().toLowerCase();

			String regex = "[a-z_\\-]{6,24}|-[\\d.,]{3,15}|[\\d.,]{4,16}";
			Pattern pattern = Pattern.compile(regex);
			Matcher matcher = pattern.matcher(line);

			while (matcher.find()) {
				String w = matcher.group();
				word.set(w);
				context.write(word, one);
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
		Job job = Job.getInstance(new Configuration(), "HadoopWordCount");
		job.setJarByClass(HadoopWordCount.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);

		job.setMapperClass(Map.class);
		job.setCombinerClass(Reduce.class);
		job.setReducerClass(Reduce.class);

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		FileInputFormat.setInputPaths(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		job.waitForCompletion(true);
		return 0;
	}

	public static void main(String[] args) throws Exception {
		int ret = ToolRunner.run(new Configuration(), new HadoopWordCount(), args);
		System.exit(ret);
	}
}