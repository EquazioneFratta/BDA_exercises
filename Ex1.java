
/* I want to implement a Hadoop MapReduce program in Java to analyze a Wikipedia collection (AA directory). 
I will provide my existing HadoopWordCount.java code that already performs tokenization according to the required regex. 
The tokenization part is already done and should NOT be modified or reimplemented. You only need to implement the MapReduce logic for counting, pairing, filtering, and top-K.
Constraints:
- Everything must be done ONLY using MapReduce (Mapper + Reducer)
- NO local post-processing is allowed
- Final results must be directly written to HDFS
- Multiple jobs are allowed, but the minimum possible (ideally 2 jobs)
- Using Combiner is allowed
- Use Hadoop MapReduce API (org.apache.hadoop.mapreduce)
-----------------------------------
TASKS TO IMPLEMENT
-----------------------------------
1) Find all words (NOT numbers) with frequency exactly = 1000
2) Find all word pairs (NOT numbers) with frequency = 1000
   - Pairs are generated with a distance of m ≤ 2 (window: i+1 and i+2)
3) Find the top-100 most frequent words
4) Find the top-100 pairs where the first token is a number and the second token is a word
   - Also using a distance of m ≤ 2
-----------------------------------
ARCHITECTURE
-----------------------------------
Implement ONLY 2 MapReduce jobs:
-----------------------------------
JOB 1: COUNTING
-----------------------------------
Mapper:
- Receives tokenized input (use the existing tokenization from my code)
- Save tokens in an ordered list
- Distinguish between words and numbers
Emit:
1) Unigrams (words only)
   KEY: "W|word"
   VALUE: 1
2) Word-word pairs (distance ≤ 2)
   KEY: "P|word1 word2"
   VALUE: 1
3) Number → word pairs (distance ≤ 2)
   KEY: "NW|number word"
   VALUE: 1
Reducer:
- Sum the values
- Output: KEY = type|content, VALUE = total count
Use Combiner for efficiency.
-----------------------------------
JOB 2: FILTER + TOP-K
-----------------------------------
Input = output of JOB 1
Mapper:
- Parse KEY into type (W, P, NW) and content
- VALUE = count
Emit:
(1) Words with count = 1000
   KEY: "FREQ1000_WORD"
   VALUE: word
(2) Word-word pairs with count = 1000
   KEY: "FREQ1000_PAIR"
   VALUE: word1 word2
(3) For top-100 words
   KEY: "TOP_WORD"
   VALUE: "word:count"
(4) For top-100 number→word pairs
   KEY: "TOP_NUMWORD"
   VALUE: "number word:count"
Reducer:
- Use a single reduce task (setNumReduceTasks(1))
For FREQ1000_* → print all values directly
For TOP_WORD and TOP_NUMWORD → use a PriorityQueue (min-heap) to maintain top 100
- Sort by count in descending order
-----------------------------------
REQUIREMENTS
-----------------------------------
- NO sorting or filtering outside MapReduce
- Top-100 must be calculated in the reducer
- Provide complete, compilable Java code
- Include:
  - Mapper and Reducer classes for both jobs
  - Driver that runs the two jobs sequentially
  - Input/output path handling
-----------------------------------
ADDITIONAL NOTE
-----------------------------------
- Modify directly the code attached that already implements the WordCount
 */
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
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

public class Ex1 extends Configured implements Tool {

    // =========================================================================
    // JOB 1 — COUNTING
    // =========================================================================

    /**
     * Job1Mapper
     *
     * Tokenizes each line (using the original regex, unchanged) and emits:
     * "W|word" → 1 for every word token
     * "P|w1 w2" → 1 for every word-word pair at distance ≤ 2
     * "NW|num word" → 1 for every number→word pair at distance ≤ 2
     *
     * Numbers are NOT emitted as unigrams (tasks 1-3 involve words only for
     * unigrams; numbers appear only as the first token in NW pairs).
     */
    public static class Job1Mapper extends Mapper<LongWritable, Text, Text, IntWritable> {

        private static final IntWritable ONE = new IntWritable(1);

        // Original regex — NOT modified
        private static final String REGEX = "[a-z_\\-]{6,24}|-[\\d.,]{3,15}|[\\d.,]{4,16}";
        private static final Pattern PATTERN = Pattern.compile(REGEX);

        // A token is a word if it matches only [a-z_\-]{6,24}
        private static final Pattern WORD_PATTERN = Pattern.compile("^[a-z_\\-]{6,24}$");
        // A token is a number if it matches the number branches of the regex
        private static final Pattern NUM_PATTERN = Pattern.compile("^-?[\\d.,]{3,15}$");

        private final Text outKey = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString().toLowerCase();
            Matcher matcher = PATTERN.matcher(line);

            // Collect all tokens in order, tagging each as word or number
            List<String> tokens = new ArrayList<>();
            List<Boolean> isWord = new ArrayList<>();

            while (matcher.find()) {
                String tok = matcher.group();
                tokens.add(tok);
                isWord.add(WORD_PATTERN.matcher(tok).matches());
            }

            for (int i = 0; i < tokens.size(); i++) {
                String ti = tokens.get(i);
                boolean tiIsWord = isWord.get(i);

                // --- Unigram (words only) ---
                if (tiIsWord) {
                    outKey.set("W|" + ti);
                    context.write(outKey, ONE);
                }

                // --- Pairs at distance d = 1 and d = 2 ---
                for (int d = 1; d <= 2; d++) {
                    int j = i + d;
                    if (j >= tokens.size())
                        break;

                    String tj = tokens.get(j);
                    boolean tjIsWord = isWord.get(j);

                    // Word–word pair
                    if (tiIsWord && tjIsWord) {
                        outKey.set("P|" + ti + " " + tj);
                        context.write(outKey, ONE);
                    }

                    // Number → word pair
                    if (!tiIsWord && tjIsWord) {
                        outKey.set("NW|" + ti + " " + tj);
                        context.write(outKey, ONE);
                    }
                }
            }
        }
    }

    /**
     * Job1Reducer (also used as Combiner)
     *
     * Simply sums counts. Output: KEY = type|content, VALUE = total count.
     */
    public static class Job1Reducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        private final IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable v : values)
                sum += v.get();
            result.set(sum);
            context.write(key, result);
        }
    }

    // =========================================================================
    // JOB 2 — FILTER + TOP-K
    // =========================================================================

    /**
     * Job2Mapper
     *
     * Reads the output of Job 1 (tab-separated "type|content \t count") and
     * re-emits entries grouped by output category:
     *
     * FREQ1000_WORD → word (words with count == 1000)
     * FREQ1000_PAIR → "w1 w2" (word-word pairs with count == 1000)
     * TOP_WORD → "word:count" (all words, for top-100 selection)
     * TOP_NUMWORD → "num word:count" (all NW pairs, for top-100 selection)
     */
    public static class Job2Mapper extends Mapper<LongWritable, Text, Text, Text> {

        private static final int TARGET_FREQ = 1000;

        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString();
            // Job 1 output format: "type|content\tcount"
            int tab = line.lastIndexOf('\t');
            if (tab < 0)
                return;

            String typeContent = line.substring(0, tab);
            int count;
            try {
                count = Integer.parseInt(line.substring(tab + 1).trim());
            } catch (NumberFormatException e) {
                return; // skip malformed lines
            }

            int pipe = typeContent.indexOf('|');
            if (pipe < 0)
                return;

            String type = typeContent.substring(0, pipe);
            String content = typeContent.substring(pipe + 1);

            switch (type) {
                case "W":
                    // Unigram word
                    if (count == TARGET_FREQ) {
                        outKey.set("FREQ1000_WORD");
                        outVal.set(content);
                        context.write(outKey, outVal);
                    }
                    // Always feed into top-100 word heap
                    outKey.set("TOP_WORD");
                    outVal.set(content + ":" + count);
                    context.write(outKey, outVal);
                    break;

                case "P":
                    // Word–word pair
                    if (count == TARGET_FREQ) {
                        outKey.set("FREQ1000_PAIR");
                        outVal.set(content);
                        context.write(outKey, outVal);
                    }
                    break;

                case "NW":
                    // Number → word pair — always feed into top-100 NW heap
                    outKey.set("TOP_NUMWORD");
                    outVal.set(content + ":" + count);
                    context.write(outKey, outVal);
                    break;

                default:
                    // Unknown type — ignore
                    break;
            }
        }
    }

    /**
     * Job2Reducer
     *
     * Single reduce task (setNumReduceTasks(1)).
     *
     * FREQ1000_WORD / FREQ1000_PAIR → emit every value directly.
     * TOP_WORD / TOP_NUMWORD → maintain a min-heap of size 100;
     * emit the top-100 entries sorted
     * by descending count.
     */
    public static class Job2Reducer extends Reducer<Text, Text, Text, Text> {

        private static final int TOP_K = 100;

        /** Min-heap entry: (count, label) */
        private static class Entry {
            final int count;
            final String label;

            Entry(int count, String label) {
                this.count = count;
                this.label = label;
            }
        }

        // Comparator: smallest count first (min-heap → evicts least frequent)
        private static final Comparator<Entry> MIN_BY_COUNT = Comparator.comparingInt(e -> e.count);

        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            String keyStr = key.toString();

            if (keyStr.equals("FREQ1000_WORD") || keyStr.equals("FREQ1000_PAIR")) {
                // Emit every value directly
                for (Text v : values) {
                    outKey.set(keyStr);
                    outVal.set(v.toString());
                    context.write(outKey, outVal);
                }

            } else if (keyStr.equals("TOP_WORD") || keyStr.equals("TOP_NUMWORD")) {
                // Build min-heap of top-K entries
                PriorityQueue<Entry> heap = new PriorityQueue<>(TOP_K + 1, MIN_BY_COUNT);

                for (Text v : values) {
                    String s = v.toString();
                    // Format: "content:count" — split on the LAST colon
                    int lastColon = s.lastIndexOf(':');
                    if (lastColon < 0)
                        continue;
                    String label = s.substring(0, lastColon);
                    int cnt;
                    try {
                        cnt = Integer.parseInt(s.substring(lastColon + 1));
                    } catch (NumberFormatException e) {
                        continue;
                    }

                    heap.offer(new Entry(cnt, label));
                    if (heap.size() > TOP_K)
                        heap.poll(); // evict minimum
                }

                // Drain heap into a list and sort descending
                List<Entry> topList = new ArrayList<>(heap);
                topList.sort((a, b) -> Integer.compare(b.count, a.count));

                int rank = 1;
                for (Entry e : topList) {
                    outKey.set(keyStr + "_RANK_" + rank);
                    outVal.set(e.label + "\t" + e.count);
                    context.write(outKey, outVal);
                    rank++;
                }
            }
        }
    }

    // =========================================================================
    // DRIVER
    // =========================================================================

    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: HadoopWordCount <inputPath> <intermediatePath> <outputPath>");
            return 1;
        }

        Path inputPath = new Path(args[0]);
        Path intermediatePath = new Path(args[1]);
        Path outputPath = new Path(args[2]);
        // ----- JOB 1 : Counting -----
        Job job1 = Job.getInstance(new Configuration(), "HadoopWordCount-Job1-Counting");
        job1.setJarByClass(Ex1.class);

        job1.setMapperClass(Job1Mapper.class);
        job1.setCombinerClass(Job1Reducer.class); // Combiner = same logic as Reducer
        job1.setReducerClass(Job1Reducer.class);

        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(IntWritable.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);

        job1.setInputFormatClass(TextInputFormat.class);
        job1.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.setInputPaths(job1, inputPath);
        FileOutputFormat.setOutputPath(job1, intermediatePath);

        boolean job1Success = job1.waitForCompletion(true);
        if (!job1Success) {
            System.err.println("Job 1 failed.");
            return 1;
        }

        // ----- JOB 2 : Filter + Top-K -----
        Job job2 = Job.getInstance(new Configuration(), "HadoopWordCount-Job2-FilterTopK");
        job2.setJarByClass(Ex1.class);

        job2.setMapperClass(Job2Mapper.class);
        job2.setReducerClass(Job2Reducer.class);
        job2.setNumReduceTasks(1); // single reducer → global top-K

        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);

        job2.setInputFormatClass(TextInputFormat.class);
        job2.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.setInputPaths(job2, intermediatePath);
        FileOutputFormat.setOutputPath(job2, outputPath);

        boolean job2Success = job2.waitForCompletion(true);
        if (!job2Success) {
            System.err.println("Job 2 failed.");
            return 1;
        }
        return 0;
    }

    public static void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new Configuration(), new Ex1(), args);
        System.exit(ret);
    }
}