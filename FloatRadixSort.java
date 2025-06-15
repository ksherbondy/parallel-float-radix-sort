import java.util.Arrays;
import java.util.concurrent.*;

public class FloatRadixSort {

    public static void parallelRadixSort(float[] floats, int numThreads) {
        int n = floats.length;
        int[] ints = new int[n];
        for (int i = 0; i < n; i++) {
            ints[i] = floatToSortableInt(floats[i]);
        }

        int[] auxInts = new int[n];
        float[] auxFloats = new float[n];

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        int chunkSize = (n + numThreads - 1) / numThreads;

        for (int pass = 0; pass < 4; pass++) {
            int shift = pass * 8;
            int[][] localCounts = new int[numThreads][256];

            // Step 1: Local histograms
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
                });
            }
            waitFor(executor, numThreads);

            // Step 2: Merge histograms
            int[] globalCount = new int[256];
            for (int i = 0; i < 256; i++) {
                for (int t = 0; t < numThreads; t++) {
                    globalCount[i] += localCounts[t][i];
                }
            }

            // Step 3: Prefix sum
            int[] startPos = new int[256];
            for (int i = 1; i < 256; i++) {
                startPos[i] = startPos[i - 1] + globalCount[i - 1];
            }

            // Step 4: Redistribute
            int[] writePositions = Arrays.copyOf(startPos, 256);
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
                });
            }
            waitFor(executor, numThreads);

            System.arraycopy(auxInts, 0, ints, 0, n);
            System.arraycopy(auxFloats, 0, floats, 0, n);
        }

        executor.shutdown();
    }

    private static void waitFor(ExecutorService executor, int tasks) {
        try {
            executor.shutdown();
            executor.awaitTermination(1, TimeUnit.HOURS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static int floatToSortableInt(float f) {
        int bits = Float.floatToIntBits(f);
        return bits ^ ((bits >> 31) | 0x80000000);
    }

    public static void main(String[] args) {
        int size = 50_000_000;
        int numThreads = Runtime.getRuntime().availableProcessors();
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = (float) (Math.random() * 1_000_000.0 - 500_000.0);
        }
        long start = System.currentTimeMillis();
        parallelRadixSort(data, numThreads);
        long end = System.currentTimeMillis();
        System.out.println("Parallel radix sort time: " + (end - start) + " ms");
    }
}
