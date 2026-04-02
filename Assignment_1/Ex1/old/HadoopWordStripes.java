import java.io.IOException;
import java.util.Iterator;

import javax.naming.Context;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapWritable;
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

public class HadoopWordStripes extends Configured implements Tool {

	private final static IntWritable one = new IntWritable(1);

	public static class Stripe extends MapWritable {

		public void add(String w) {
			IntWritable count = new IntWritable(0);
			if (containsKey(new Text(w))) {
				count = (IntWritable) get(new Text(w));
				remove(new Text(w));
			}
			count = new IntWritable(count.get() + one.get());
			put(new Text(w), count);
		}

	    public void merge(Stripe from) {
	        for (Writable fromKey : from.keySet())
	            if (containsKey(fromKey))
	                put(fromKey, new IntWritable(((IntWritable) get(fromKey)).get() + ((IntWritable) from.get(fromKey)).get()));
	            else
	                put(fromKey, (IntWritable) from.get(fromKey));
	    }

	    @Override
	    public String toString() {
	        StringBuilder buffer = new StringBuilder();
	        for (Writable key : keySet())
	            buffer.append(" ").append(key).append(":").append(get(key));
	        return buffer.toString();
	    }
	}

	public static class Map extends Mapper<LongWritable, Text, Text, Stripe> {

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
			for (int i = 0; i < splitLine.length; i++) {
				
				Stripe stripe = new Stripe();
				String w;

				if (i > 0) {
					w = splitLine[i - 1];
					stripe.add(w);
				}

				if (i < splitLine.length - 1) {
					w = splitLine[i + 1];
					stripe.add(w);
				}

				context.write(new Text(splitLine[i]), stripe);
			}
		}
	}

	public static class Reduce extends Reducer<Text, Stripe, Text, Stripe> {

		@Override
		public void reduce(Text key, Iterable<Stripe> values, Context context)
				throws IOException, InterruptedException {
			
			Stripe globalStripe = new Stripe();

			for (Stripe localStripe : values)
				globalStripe.merge(localStripe);
			
			context.write(key, globalStripe);
		}
	}

	@Override
	public int run(String[] args) throws Exception {
		Job job = Job.getInstance(new Configuration(), "HadoopWordStripes");
		job.setJarByClass(HadoopWordStripes.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Stripe.class);

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
		int ret = ToolRunner.run(new Configuration(), new HadoopWordStripes(), args);
		System.exit(ret);
	}
}
