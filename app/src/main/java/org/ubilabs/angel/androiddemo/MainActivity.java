package org.ubilabs.angel.androiddemo;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Handler;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.ubilabs.angel.uitl.PermissionUtils;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static org.opencv.imgproc.Imgproc.MORPH_RECT;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    /*
    * Initial & Debug
    * */
    private static final String TAG = "MainActivity";

    private SeekBar threshold1;
    private TextView threshold1Dislpay;
    private TextView displayNumber;
    private String numberString = "-1";
    private Handler handler;

    @SuppressWarnings("SuspiciousNameCombination")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        handler = new Handler();

        initDebug();
        initTensorFlowAndLoadModel();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
                || ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
                || ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermission();
        } else {
            initCamera();
        }
    }

    private void initDebug() {
        threshold1 = findViewById(R.id.threshold1);
        threshold1Dislpay = findViewById(R.id.threshold1Display);
        threshold1.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                threshold1Dislpay.setText(String.valueOf(i));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });
        threshold1.setProgress(24);
        displayNumber = findViewById(R.id.displayNumber);
    }

    /**
     * Callback received when a permissions request has been completed.
     */
    @Override
    public void onRequestPermissionsResult(final int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        PermissionUtils.requestPermissionsResult(this, requestCode, permissions, grantResults, mPermissionGrant);
        initCamera();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (openCvCameraView != null) {
            openCvCameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        initCamera();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (openCvCameraView != null) {
            openCvCameraView.disableView();
        }
    }

    /*
    * OpenCV
    * */
    private CameraBridgeViewBase openCvCameraView;

    static {
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV not loaded");
        } else {
            Log.d(TAG, "OpenCV loaded");
        }
    }

    private Mat tmpMat;
    private Mat emptyMat;
    private Mat kernelDilate;
    private Mat kernelErode;
    private static final int MAXWIDTH = 320;
    private static final int MAXHEIGHT = 240;
    private static final int IMGSIZE = 28;

    private void initCamera() {
        openCvCameraView = findViewById(R.id.HelloOpenCvView);
        openCvCameraView.setVisibility(SurfaceView.VISIBLE);
        openCvCameraView.setCvCameraViewListener(this);
        openCvCameraView.setMaxFrameSize(MAXWIDTH, MAXHEIGHT);
        openCvCameraView.enableFpsMeter();
        openCvCameraView.enableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        tmpMat = new Mat();
        emptyMat = new Mat();
        kernelDilate = Imgproc.getStructuringElement(MORPH_RECT, new Size(2, 2));
        kernelErode = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(1, 1));
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat grayImg = inputFrame.gray();
        Mat cannyImg = tmpMat;
        emptyMat.copyTo(cannyImg);

        int ratio = 3;
        Imgproc.Canny(grayImg, cannyImg, threshold1.getProgress(), threshold1.getProgress() * ratio);
        Imgproc.dilate(cannyImg, cannyImg, kernelDilate);
        Imgproc.erode(cannyImg, cannyImg, kernelErode);

        Mat numBerImg = new Mat(cannyImg, new Rect((MAXWIDTH - IMGSIZE) / 2, (MAXHEIGHT - IMGSIZE) / 2, IMGSIZE, IMGSIZE));
        doRecognize(numBerImg);

        Core.rectangle(cannyImg, new Point((MAXWIDTH - IMGSIZE) / 2, (MAXHEIGHT - IMGSIZE) / 2), new Point((MAXWIDTH + IMGSIZE) / 2, (MAXHEIGHT + IMGSIZE) / 2), new Scalar(255), 1);
        return cannyImg;
    }

    private void requestPermission() {
        PermissionUtils.requestMultiPermissions(this, mPermissionGrant);
    }

    private PermissionUtils.PermissionGrant mPermissionGrant = new PermissionUtils.PermissionGrant() {
        @Override
        public void onPermissionGranted(int requestCode) {
            switch (requestCode) {
                case PermissionUtils.CODE_CAMERA:
                    Toast.makeText(MainActivity.this, "Result Permission Grant CODE_CAMERA", Toast.LENGTH_SHORT).show();
                    break;
                case PermissionUtils.CODE_READ_EXTERNAL_STORAGE:
                    Toast.makeText(MainActivity.this, "Result Permission Grant CODE_READ_EXTERNAL_STORAGE", Toast.LENGTH_SHORT).show();
                    break;
                case PermissionUtils.CODE_WRITE_EXTERNAL_STORAGE:
                    Toast.makeText(MainActivity.this, "Result Permission Grant CODE_WRITE_EXTERNAL_STORAGE", Toast.LENGTH_SHORT).show();
                    break;
                default:
                    Toast.makeText(MainActivity.this, "Result Permission Grant CODE_MULTI_PERMISSION", Toast.LENGTH_SHORT).show();
                    break;
            }
        }
    };

    /*
    * TensorFlow
    * */

    //TODO Declare input model variables here
    private static final int INPUT_SIZE = 28;
    private static final String INPUT_NAME = "";
    private static final String OUTPUT_NAME = "";

    private static final String MODEL_FILE = "";

    private TensorFlowImageClassifier classifier;
    private Executor executor = Executors.newSingleThreadExecutor();

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorFlowImageClassifier.create(
                            getAssets(),
                            MODEL_FILE,
                            INPUT_SIZE,
                            INPUT_NAME,
                            OUTPUT_NAME);
                    Log.d(TAG, "Load Success");
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private void doRecognize(Mat input) {
        float[] pixels = new float[INPUT_SIZE * INPUT_SIZE];

        int cnt = 0;
        for (int i = 0; i < input.rows(); i++) {
            for (int j = 0; j < input.cols(); j++) {
                pixels[cnt] = (float) input.get(i, j)[0] * 1f / 255;
                cnt++;
            }
        }

        final int results = classifier.recognizeImage(pixels);

        numberString = " Number is : " + results;
        Log.d(TAG, numberString);
        handler.post(updateView);
    }

    Runnable updateView = new Runnable() {

        @Override
        public void run() {
            displayNumber.setText(numberString);
        }

    };
}
