package org.ubilabs.angel;

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
import org.ubilabs.angel.mnist.R;
import org.ubilabs.angel.model.TensorFlowLiteDetector;
import org.ubilabs.angel.uitl.PermissionUtils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opencv.imgproc.Imgproc.MORPH_RECT;
import static org.ubilabs.angel.setting.ImageSetting.MAXHEIGHT;
import static org.ubilabs.angel.setting.ImageSetting.MAXWIDTH;
import static org.ubilabs.angel.setting.TensorFlowSetting.DIM_IMG_SIZE_X;
import static org.ubilabs.angel.setting.TensorFlowSetting.DIM_IMG_SIZE_Y;

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

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
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
        kernelErode = Imgproc.getStructuringElement(MORPH_RECT, new Size(1, 1));
        kernelDilate = Imgproc.getStructuringElement(MORPH_RECT, new Size(2, 2));

        initModel();
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
        Imgproc.erode(cannyImg, cannyImg, kernelErode);
        Imgproc.dilate(cannyImg, cannyImg, kernelDilate);

        Mat numBerImg = new Mat(cannyImg, new Rect((MAXWIDTH - DIM_IMG_SIZE_X) / 2, (MAXHEIGHT - DIM_IMG_SIZE_Y) / 2, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y));

        List<TensorFlowLiteDetector.Recognition> results = detector.detectImage(numBerImg);
        if (results != null) {
            Log.e(TAG,String.valueOf(results));
            numberString = " Number is : " + results.get(0).getTitle();
            Log.d(TAG, numberString);
            handler.post(updateView);
        }

        Core.rectangle(cannyImg, new Point((MAXWIDTH - DIM_IMG_SIZE_X) / 2, (MAXHEIGHT - DIM_IMG_SIZE_Y) / 2), new Point((MAXWIDTH + DIM_IMG_SIZE_X) / 2, (MAXHEIGHT + DIM_IMG_SIZE_Y) / 2), new Scalar(255), 1);
        return cannyImg;
    }

    private void requestPermission() {
        PermissionUtils.requestMultiPermissions(this, mPermissionGrant);
    }

    private PermissionUtils.PermissionGrant mPermissionGrant = requestCode -> {
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
    };

    /*
     * TensorFlow
     * */
    private TensorFlowLiteDetector detector;

    private void initModel() {
        Map<String, Object> othersMap = new HashMap<>();
        othersMap.put("activity", this);

        detector = new TensorFlowLiteDetector(othersMap);
    }

    Runnable updateView = new Runnable() {
        @Override
        public void run() {
            displayNumber.setText(numberString);
        }

    };
}
