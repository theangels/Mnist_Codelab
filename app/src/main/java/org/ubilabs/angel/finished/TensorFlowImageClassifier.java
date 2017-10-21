package org.ubilabs.angel.finished;

import android.content.res.AssetManager;
import android.support.v4.os.TraceCompat;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;
import java.util.Arrays;


class TensorFlowImageClassifier {

    private static final String TAG = "TFImageClassifier";

    // Config values.
    private String inputName;
    private String outputName;
    private int inputSize;

    // Pre-allocated buffers.
    private float[] outputs;

    private TensorFlowInferenceInterface inferenceInterface;

    private TensorFlowImageClassifier() {
    }

    public static TensorFlowImageClassifier create(
            AssetManager assetManager,
            String modelFilename,
            int inputSize,
            String inputName,
            String outputName)
            throws IOException {
        TensorFlowImageClassifier c = new TensorFlowImageClassifier();
        c.inputName = inputName;
        c.outputName = outputName;

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        int numClasses = (int) c.inferenceInterface.graph().operation(outputName).output(0).shape().size(1);
        Log.i(TAG, "Output layer size is " + numClasses);

        c.inputSize = inputSize;
        c.outputName = outputName;
        c.outputs = new float[numClasses];

        return c;
    }

    int recognizeImage(final float[] pixels) {
        int recognitions = 0;

        // Log this method so that it can be analyzed with systrace.
        TraceCompat.beginSection("recognizeImage");

        // Copy the input data into TensorFlow.
        TraceCompat.beginSection("feed");
        inferenceInterface.feed(inputName, pixels, inputSize * inputSize);
        TraceCompat.endSection();

        // Run the inference call.
        TraceCompat.beginSection("run");
        inferenceInterface.run(new String[]{outputName}, false);
        TraceCompat.endSection();

        // Copy the output Tensor back into the output array.
        TraceCompat.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        TraceCompat.endSection();

        Log.d(TAG, "" + Arrays.toString(outputs));


        for (int i = 1; i < outputs.length; i++) {
            if (outputs[recognitions] > outputs[i]) {
                recognitions = i;
            }
        }

        TraceCompat.endSection(); // "recognizeImage"
        return recognitions;
    }
}


