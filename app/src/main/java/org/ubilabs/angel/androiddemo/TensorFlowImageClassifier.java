package org.ubilabs.angel.androiddemo;

import java.io.IOException;

import android.content.res.AssetManager;
import android.util.Log;


class TensorFlowImageClassifier {

    private static final String TAG = "TFImageClassifier";

    // Config values.
    private String inputName;
    private String outputName;
    private int inputSize;

    // Pre-allocated buffers.
    private float[] outputs;

    //TODO Add member here

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

        int numClasses = 0;

        //TODO Add TensorFlowInferenceInterface instantiation here

        Log.i(TAG, "Output layer size is " + numClasses);

        c.inputSize = inputSize;
        c.outputName = outputName;
        c.outputs = new float[numClasses];

        return c;
    }

    int recognizeImage(final float[] pixels) {
        int recognitions = 0;

        //TODO Add TensorFlow process code here

        return recognitions;
    }
}


