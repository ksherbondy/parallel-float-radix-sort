import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class FloatRadixSort {

    public static void parallelRadixSort(float[] floats) throws InterruptedException {
        int n = floats.length;
        int[] ints = new int[n];

        for (int i = 0; i < n; i++) {
            ints[i] = floatToSortableInt(floats[i]);
        }

        int[] auxInts = new int[n];
        float[] auxFloats = new float[n];

        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        int chunkSize = (n + numThreads - 1) / numThreads;  // ceiling division

        for (int pass = 0; pass < 4; pass++) {
            int shift = pass * 8;
            int[][] localCounts = new int[numThreads][256];

            CountDownLatch latch1 = new CountDownLatch(numThreads);
            // Counting phase (each thread counts frequency of byte values)
            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    int start = threadId * chunkSize;
                    int end = Math.min(start + chunkSize, n);
                    int[] count = localCounts[threadId];
                    for (int i = start; i < end; i++) {
                        int c = (ints[i] >>> shift) & 0xFF;
                        count[c]++;
                    }
                    latch1.countDown();
                });
            }
            latch1.await();

            // Aggregate counts globally
            int[] globalCount = new int[256];
            for (int i = 0; i < 256; i++) {
                for (int t = 0; t < numThreads; t++) {
                    globalCount[i] += localCounts[t][i];
                }
            }

            // Compute start positions
            int[] startPos = new int[256];
            for (int i = 1; i < 256; i++) {
                startPos[i] = startPos[i - 1] + globalCount[i - 1];
            }

            // Copy start positions for threads
            int[] writePositions = Arrays.copyOf(startPos, 256);

            CountDownLatch latch2 = new CountDownLatch(numThreads);
            // Distribution phase (each thread moves elements to output array)
            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                executor.submit(() -> {
                    int start = threadId * chunkSize;
                    int end = Math.min(start + chunkSize, n);
                    int[] localPos = new int[256];

                    synchronized (writePositions) {
                        for (int i = 0; i < 256; i++) {
                            localPos[i] = writePositions[i];
                            writePositions[i] += localCounts[threadId][i];
                        }
                    }

                    for (int i = start; i < end; i++) {
                        int c = (ints[i] >>> shift) & 0xFF;
                        int pos = localPos[c]++;
                        auxInts[pos] = ints[i];
                        auxFloats[pos] = floats[i];
                    }
                    latch2.countDown();
                });
            }
            latch2.await();

            System.arraycopy(auxInts, 0, ints, 0, n);
            System.arraycopy(auxFloats, 0, floats, 0, n);
        }

        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);
    }

    private static int floatToSortableInt(float f) {
        int bits = Float.floatToIntBits(f);
        return bits ^ ((bits >> 31) | 0x80000000);
    }

    public static void main(String[] args) throws InterruptedException {
        final int SIZE = 50_000_000;
        Random rand = new Random(42);

        System.out.println("Generating " + SIZE + " random floats...");
        float[] original = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            original[i] = rand.nextFloat() * 1_000_000f - 500_000f;
        }

        // Make copies for both sorts
        float[] arrJavaSort = Arrays.copyOf(original, original.length);
        float[] arrParallelRadix = Arrays.copyOf(original, original.length);

        System.out.println("\nSorting with Arrays.sort...");
        long start = System.currentTimeMillis();
        Arrays.sort(arrJavaSort);
        long end = System.currentTimeMillis();
        System.out.println("Arrays.sort time: " + (end - start) + " ms");

        System.out.println("\nSorting with parallelRadixSort...");
        start = System.currentTimeMillis();
        parallelRadixSort(arrParallelRadix);
        end = System.currentTimeMillis();
        System.out.println("Parallel radix sort time: " + (end - start) + " ms");

        System.out.println("\nVerifying results...");
        if (Arrays.equals(arrJavaSort, arrParallelRadix)) {
            System.out.println("Sorting correctness verified ✅");
        } else {
            System.out.println("Sorting correctness FAILED ❌");
        }
    }
}
