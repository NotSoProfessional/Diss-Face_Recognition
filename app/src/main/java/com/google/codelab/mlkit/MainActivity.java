// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.codelab.mlkit;

/*import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC4;
import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imdecode;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_RGBA2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;*/

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.hardware.Camera;
import android.os.Build;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.os.SystemClock;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.webkit.ConsoleMessage;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.codelab.mlkit.env.Logger;
import com.google.codelab.mlkit.tflite.SimilarityClassifier;
import com.google.codelab.mlkit.tflite.TFLiteObjectDetectionAPIModel;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.linear.*;
/*//import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_face;
import org.bytedeco.opencv.opencv_face.FisherFaceRecognizer;*/
//import org.opencv.android.OpenCVLoader;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_java;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.face.EigenFaceRecognizer;
import org.opencv.face.FaceRecognizer;
import org.opencv.face.FacemarkKazemi;
import org.opencv.face.FacemarkLBF;
import org.opencv.face.FisherFaceRecognizer;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.tensorflow.lite.TensorFlowLite;


public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {
    private static final String TAG = "MainActivity";
    private static final Logger LOGGER = new Logger();
    private ImageView mImageView;
    private ImageView mImageViewProcessed;
    private Button mFaceButton;
    private Button mTrainButton;
    private Switch mGsSwitch, mRotSwitch, mEqSwitch, mCVSwitch, mNNAPISwitch, mGPUSwitch, mXNNSwitch;
    private EditText mNThreads;
    private Bitmap mSelectedImage;
    private GraphicOverlay mGraphicOverlay;
    // Max width (portrait mode)
    private Integer mImageMaxWidth;
    // Max height (portrait mode)
    private Integer mImageMaxHeight;

    private CVFaceRecognizer cvFR;
    //private MatVector mv = new MatVector();
    private ArrayList<Integer> labels = new ArrayList<Integer>();


    private static final int TF_OD_API_INPUT_SIZE = 112;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "mobile_face_net.tflite";


    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;

    private SimilarityClassifier detector;

    private static String[] assets;
    private static final String image_path = "images_dev";

    private static boolean train = false;

    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mImageView = findViewById(R.id.image_view);
        mImageViewProcessed = findViewById(R.id.image_view_processed);

        mGsSwitch = findViewById(R.id.gs_switch);
        mRotSwitch = findViewById(R.id.rot_switch);
        mEqSwitch = findViewById(R.id.eq_switch);
        mCVSwitch = findViewById(R.id.cv_switch);
        mNNAPISwitch = findViewById(R.id.nnapi_switch);
        mGPUSwitch = findViewById(R.id.gpu_switch);
        mXNNSwitch = findViewById(R.id.xnnpack_switch);

        mNThreads = findViewById(R.id.nthreads_input);

        mFaceButton = findViewById(R.id.button_face);

        mTrainButton = findViewById(R.id.button_train);

        mGraphicOverlay = findViewById(R.id.graphic_overlay);

        Log.d("NSP debug: ", TensorFlowLite.schemaVersion());
        Log.d("NSP debug: ", TensorFlowLite.runtimeVersion());

        mNNAPISwitch.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (mNNAPISwitch.isChecked()){
                    mGPUSwitch.setEnabled(false);
                    mXNNSwitch.setEnabled(false);

                }else{
                    mGPUSwitch.setEnabled(true);
                    mXNNSwitch.setEnabled(true);
                }

                useDelegates(mNNAPISwitch.isChecked(), mGPUSwitch.isChecked(), mXNNSwitch.isChecked());
            }
        });

        mGPUSwitch.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (mGPUSwitch.isChecked()){

                    mNNAPISwitch.setEnabled(false);
                    mXNNSwitch.setEnabled(false);
                } else{
                    mNNAPISwitch.setEnabled(true);
                    mXNNSwitch.setEnabled(true);
                }

                useDelegates(mNNAPISwitch.isChecked(), mGPUSwitch.isChecked(), mXNNSwitch.isChecked());
            }
        });

        mXNNSwitch.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (mXNNSwitch.isChecked()){
                    mGPUSwitch.setEnabled(false);
                    mNNAPISwitch.setEnabled(false);

                }else{
                    mGPUSwitch.setEnabled(true);
                    mNNAPISwitch.setEnabled(true);
                }

                useDelegates(mNNAPISwitch.isChecked(), mGPUSwitch.isChecked(), mXNNSwitch.isChecked());
            }
        });

        mFaceButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                runFaceContourDetection();

            }
        });

        mTrainButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                trainFace();
            }
        });

        try {
            assets = getAssets().list(image_path);
        } catch (final IOException e) {
            e.printStackTrace();
        }

        Spinner dropdown = findViewById(R.id.spinner);

        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout
                .simple_spinner_dropdown_item, assets);
        dropdown.setAdapter(adapter);
        dropdown.setSelection(0);
        dropdown.setOnItemSelectedListener(this);

        /*if (OpenCVLoader.initDebug()){
            Log.d("LOADED", "OpenCV success");
        }else Log.d("LOADED", "OpenCV error");*/

        Loader.load(opencv_java.class);
        org.bytedeco.opencv.global.opencv_core.setUseOpenCL(true);
        org.opencv.core.Core.setUseIPP(true);
        org.opencv.core.Core.setUseIPP_NotExact(true);
        Log.d("NSP debug", "OpenCV threads: " + Integer.toString(org.opencv.core.Core.getNumThreads()));
        org.opencv.core.Core.setNumThreads(4);
        //Log.d("NSP debug", "OpenCV threads: " + Integer.toString(org.opencv.core.Core.getNumThreads()));
        Log.d("NSP debug", "OpenCV OpenCL available: " + opencv_core.haveOpenCL());
        cvFR = new CVFaceRecognizer(CVFaceRecognizer.Method.LBPH);

       TensorFlowLite.init();

        try {
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED,
                            mNNAPISwitch.isChecked(),
                            mGPUSwitch.isChecked(),
                            mXNNSwitch.isChecked()
                            );
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }
    }

    private void useDelegates(boolean useNNAPI, boolean useGPU, boolean useXNNPack){
        detector.ReInitModel(getAssets(),
                TF_OD_API_MODEL_FILE,
                useNNAPI,
                useGPU,
                useXNNPack);
    }

    private void trainFace(){
        train = true;
        runFaceContourDetection();
    }

    private void runFaceContourDetection() {
        InputImage image = InputImage.fromBitmap(mSelectedImage, 0);
        FaceDetectorOptions options =
                new FaceDetectorOptions.Builder()
                        //.setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                        //.setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                        //.setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                        .build();

        mFaceButton.setEnabled(false);
        mTrainButton.setEnabled(false);

        FaceDetector detector = FaceDetection.getClient(options);
        detector.process(image)
                .addOnSuccessListener(
                        new OnSuccessListener<List<Face>>() {
                            @Override
                            public void onSuccess(List<Face> faces) {
                                new Thread(new Runnable() {
                                    @Override
                                    public void run() {
                                        processFaceContourDetectionResult(faces);
                                    }
                                }).start();
                                //processFaceContourDetectionResult(faces);

                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                // Task failed with an exception
                                mFaceButton.setEnabled(true);
                                mTrainButton.setEnabled(true);
                                e.printStackTrace();
                            }
                        });
    }

    boolean untrained = true;

    private void processFaceContourDetectionResult(List<Face> faces) {
        detector.setNumThreads(4);

        if (mGPUSwitch.isChecked()){
            detector.ReInitModel(getAssets(), TF_OD_API_MODEL_FILE, false, true, false);
        }
        // Task completed successfully
        if (faces.size() == 0) {
            showToast("No face found");
            return;
        }

        ArrayList<Bitmap> processed_faces = new ArrayList<>();

        //ArrayList<Thread> ts = new ArrayList<>();

        mGraphicOverlay.clear();

        ImagePreProcessor.Options ppOptions = new ImagePreProcessor.Options();
        ppOptions.setEqualise(mEqSwitch.isChecked());
        ppOptions.setGrayscale(mGsSwitch.isChecked());
        ppOptions.setRotate(mRotSwitch.isChecked());
        ppOptions.setUseOpenCV(mCVSwitch.isChecked());

        ImagePreProcessor imgProcessor = new ImagePreProcessor(ppOptions);

        for (Face face : faces) {
            //Thread t = new Thread(new Runnable() {
            new Thread(new Runnable() {
                @Override
                public void run() {

                    Bitmap output = imgProcessor.Process(face, mSelectedImage);

                    processed_faces.add(output);

            mImageViewProcessed.post(new Runnable() {
                @Override
                public void run() {
                    mImageViewProcessed.setImageBitmap(output);
                }
            });

                }
            }).run();
        }


        while (processed_faces.size() < faces.size());



        /*//MatVector mv = new MatVector();

        if ( train){
            for (Bitmap face : processed_faces){
                Mat mat = new Mat(face.getHeight(), face.getWidth(), CV_8UC4);

// Convert Bitmap to Mat manually
                ByteBuffer byteBuffer = ByteBuffer.allocate(face.getByteCount());
                face.copyPixelsToBuffer(byteBuffer);
                byteBuffer.position(0);
                byte[] byteArray = new byte[byteBuffer.remaining()];
                byteBuffer.get(byteArray);
                mat.data().put(byteArray);

// Convert the Mat to grayscale
                cvtColor(mat, mat, CV_RGBA2GRAY);

                mv.push_back(mat);
            }
        }*/




        for (int i = 0; i < processed_faces.size(); i++){
            Bitmap resized = processed_faces.get(i);
            FaceContourGraphic faceGraphic = new FaceContourGraphic(mGraphicOverlay);
            Face face = faces.get(i);

            List<SimilarityClassifier.Recognition> results = detector.recognizeImage(resized, false);

            if (train) {
                Log.d("NSP debug: ", "Registering face!");

                int randomNum = -1;
                final String id;

                randomNum = ThreadLocalRandom.current().nextInt(0, 1000 + 1);

                id = Integer.toString(randomNum);

                labels.add(randomNum);

                cvFR.addFace(resized, randomNum);

                detector.register(id, detector.recognizeImage(resized, true).get(0));
                faceGraphic.setId(id);

                MainActivity.super.runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        mGraphicOverlay.add(faceGraphic);
                        faceGraphic.updateFace(face);
                        faceGraphic.setId(id);
                    }
                });


            } else {
               /* IntPointer predictedLabel = new IntPointer(1);
                DoublePointer predictedConfidence = new DoublePointer(1);

                Mat mat = new Mat(resized.getHeight(), resized.getWidth(), CV_8UC4);

// Convert Bitmap to Mat manually
                ByteBuffer byteBuffer = ByteBuffer.allocate(resized.getByteCount());
                resized.copyPixelsToBuffer(byteBuffer);
                byteBuffer.position(0);
                byte[] byteArray = new byte[byteBuffer.remaining()];
                byteBuffer.get(byteArray);
                mat.data().put(byteArray);

                cvtColor(mat, mat, CV_RGBA2GRAY);*/

                long startTime = System.nanoTime();
                CVFaceRecognizer.Result result = cvFR.predict(resized);
                long endTime = System.nanoTime();

                long elapsedTime = endTime - startTime;

                Log.d("NSP debug", "Fisherface: P: " + Integer.toString(result.label) + " Confidence: "+ Double.toString(result.confidence) + " Predict time: " + elapsedTime);

                for (SimilarityClassifier.Recognition rec : results){
                    float distance = rec.getDistance();
                    String title = rec.getTitle();

                    if (distance < 1.1){
                        Log.d("NSP debug: ", rec.getDistance().toString());
                        Log.d("NSP debug: ", rec.getTitle());
                        faceGraphic.setId(rec.getTitle());

                        MainActivity.super.runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                mGraphicOverlay.add(faceGraphic);
                                faceGraphic.updateFace(face);
                                faceGraphic.setId(rec.getTitle());
                                //train = false;
                            }
                        });
                    }else{
                        Log.d("NSP debug: ", "No similar face found!");

                        MainActivity.super.runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                mGraphicOverlay.add(faceGraphic);
                                faceGraphic.updateFace(face);
                                faceGraphic.setId("Not recognised!");
                            }
                        });
                    }
                }

                Long infer_time = detector.getInferenceTime();
                Log.d("NSP debug: ", Long.toString(infer_time) + " [ns]");

            }
        }

        if (train){
            /*int[] labelsArray = new int[labels.size()];

            for (int l = 0; l < labels.size(); l++){
                labelsArray[l] = labels.get(l);
            }

            IntPointer labelsPointer = new IntPointer(labelsArray.length);

            // Set the data in the IntPointer
            labelsPointer.put(labelsArray);

            // Create a Mat object from the IntPointer
            Mat labelsMat = new Mat(labelsArray.length, 1, CV_32SC1, new Pointer(labelsPointer));

            ff.train(mv, labelsMat);*/

            cvFR.train();

        }


        //while (mGraphicOverlay.getNumGraphics() < faces.size()) {

       // }

        train = false;

        MainActivity.super.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mFaceButton.setEnabled(true);
                mTrainButton.setEnabled(true);
            }
        });
    }

    private void showToast(String message) {
        Toast.makeText(getApplicationContext(), message, Toast.LENGTH_SHORT).show();
    }

    // Functions for loading images from app assets.

    // Returns max image width, always for portrait mode. Caller needs to swap width / height for
    // landscape mode.
    private Integer getImageMaxWidth() {
        if (mImageMaxWidth == null) {
            // Calculate the max width in portrait mode. This is done lazily since we need to
            // wait for
            // a UI layout pass to get the right values. So delay it to first time image
            // rendering time.
            mImageMaxWidth = mImageView.getWidth();
        }

        return mImageMaxWidth;
    }

    // Returns max image height, always for portrait mode. Caller needs to swap width / height for
    // landscape mode.
    private Integer getImageMaxHeight() {
        if (mImageMaxHeight == null) {
            // Calculate the max width in portrait mode. This is done lazily since we need to
            // wait for
            // a UI layout pass to get the right values. So delay it to first time image
            // rendering time.
            mImageMaxHeight =
                    mImageView.getHeight();
        }

        return mImageMaxHeight;
    }

    // Gets the targeted width / height.
    private Pair<Integer, Integer> getTargetedWidthHeight() {
        int targetWidth;
        int targetHeight;
        int maxWidthForPortraitMode = getImageMaxWidth();
        int maxHeightForPortraitMode = getImageMaxHeight();
        targetWidth = maxWidthForPortraitMode;
        targetHeight = maxHeightForPortraitMode;
        return new Pair<>(targetWidth, targetHeight);
    }

    public void onItemSelected(AdapterView<?> parent, View v, int position, long id) {
        mGraphicOverlay.clear();

        mSelectedImage = getBitmapFromAsset(this, image_path + "/" + assets[position]);

        if (mSelectedImage != null) {
            // Get the dimensions of the View
            Pair<Integer, Integer> targetedSize = getTargetedWidthHeight();

            int targetWidth = targetedSize.first;
            int maxHeight = targetedSize.second;

            // Determine how much to scale down the image
            float scaleFactor =
                    Math.max(
                            (float) mSelectedImage.getWidth() / (float) targetWidth,
                            (float) mSelectedImage.getHeight() / (float) maxHeight);

            Bitmap resizedBitmap =
                    Bitmap.createScaledBitmap(
                            mSelectedImage,
                            (int) (mSelectedImage.getWidth() / scaleFactor),
                            (int) (mSelectedImage.getHeight() / scaleFactor),
                            true);

            mImageView.setImageBitmap(resizedBitmap);
            mSelectedImage = resizedBitmap;
        }
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {
        // Do nothing
    }

    public static Bitmap getBitmapFromAsset(Context context, String filePath) {
        AssetManager assetManager = context.getAssets();

        InputStream is;
        Bitmap bitmap = null;
        try {
            is = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(is);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return bitmap;
    }
}
