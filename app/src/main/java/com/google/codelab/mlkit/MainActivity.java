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
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
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
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.linear.*;
public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {
    private static final String TAG = "MainActivity";
    private static final Logger LOGGER = new Logger();
    private ImageView mImageView;
    private ImageView mImageViewProcessed;
    private Button mFaceButton;
    private Button mTrainButton;
    private Switch mGsSwitch;
    private Switch mRotSwitch;
    private Switch mEqSwitch;
    private Bitmap mSelectedImage;
    private GraphicOverlay mGraphicOverlay;
    // Max width (portrait mode)
    private Integer mImageMaxWidth;
    // Max height (portrait mode)
    private Integer mImageMaxHeight;


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

    private static boolean grayscale = true;
    private static boolean equalised = true;
    private static boolean rot_align = true;

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

        mFaceButton = findViewById(R.id.button_face);

        mTrainButton = findViewById(R.id.button_train);

        mGraphicOverlay = findViewById(R.id.graphic_overlay);

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

        try {
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED);
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
                        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                        .build();

        mFaceButton.setEnabled(false);
        FaceDetector detector = FaceDetection.getClient(options);
        detector.process(image)
                .addOnSuccessListener(
                        new OnSuccessListener<List<Face>>() {
                            @Override
                            public void onSuccess(List<Face> faces) {
                                mFaceButton.setEnabled(true);
                                processFaceContourDetectionResult(faces);
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                // Task failed with an exception
                                mFaceButton.setEnabled(true);
                                e.printStackTrace();
                            }
                        });
    }

    boolean untrained = true;

    private void processFaceContourDetectionResult(List<Face> faces) {
        // Task completed successfully
        if (faces.size() == 0) {
            showToast("No face found");
            return;
        }
        mGraphicOverlay.clear();
        for (int i = 0; i < faces.size(); ++i) {
            Face face = faces.get(i);
            String landm = String.valueOf(face.getLandmark(0));
            Log.d("NSP debug: ", landm);
            FaceContourGraphic faceGraphic = new FaceContourGraphic(mGraphicOverlay);
            mGraphicOverlay.add(faceGraphic);
            faceGraphic.updateFace(face);


            //if (i==1){
                int width = face.getBoundingBox().width();
                int height = face.getBoundingBox().height();
                float x = face.getBoundingBox().centerX();
                float y = face.getBoundingBox().centerY();
                float xOffset = width / 2.0f;
                float yOffset = height / 2.0f;
                float left = x - xOffset;
                float top = y - yOffset;

                Bitmap processed = mSelectedImage;


            if (mGsSwitch.isChecked()) {
                Bitmap bm_grayscale = Bitmap.createBitmap(mSelectedImage.getWidth(), mSelectedImage.getHeight(), Bitmap.Config.ARGB_8888);

                Canvas canvas = new Canvas(bm_grayscale);
                Paint paint = new Paint();
                ColorMatrix colorMatrix = new ColorMatrix();
                colorMatrix.setSaturation(0); // Set saturation to 0 to convert to grayscale
                paint.setColorFilter(new ColorMatrixColorFilter(colorMatrix));

                canvas.drawBitmap(processed, 0, 0, paint);

                processed = bm_grayscale;
            }

            if (mRotSwitch.isChecked()){

                Bitmap rotated = Bitmap.createBitmap(mSelectedImage.getWidth(), mSelectedImage.getHeight(), Bitmap.Config.ARGB_8888);

                Canvas canvas = new Canvas(rotated);
                Paint drawPaint = new Paint();
                drawPaint.setAntiAlias(false);
                drawPaint.setFilterBitmap(false);

                Matrix matrix = new Matrix();
                matrix.setRotate(face.getHeadEulerAngleZ(), (int) x, (int) y);
                canvas.setMatrix(matrix);

                canvas.drawBitmap(processed, 0,0, drawPaint);

                processed = rotated;

            }


            // Crop image to face
            processed = Bitmap.createBitmap(processed, (int) left, (int) top, width, height);


            if (mEqSwitch.isChecked()){
                // Compute the histogram of the grayscale image
                int[] histogram = new int[256];
                for (int x_i = 0; x_i < processed.getWidth(); x_i++) {
                    for (int y_i = 0; y_i < processed.getHeight(); y_i++) {
                        int pixel = processed.getPixel(x_i, y_i);
                        int intensity = Color.red(pixel);
                        histogram[intensity]++;
                    }
                }

                // Compute the cumulative distribution function (CDF) of the histogram
                int[] cdf = new int[256];
                cdf[0] = histogram[0];
                for (int j = 1; j < 256; j++) {
                    cdf[j] = cdf[j - 1] + histogram[j];
                }

                // Compute the mapping function for each pixel intensity value
                int numPixels = processed.getWidth() * processed.getHeight();
                int[] mapping = new int[256];
                for (int j = 0; j < 256; j++) {
                    float ratio = (float) cdf[j] / numPixels;
                    mapping[j] = (int) (ratio * 255);
                }

                // Apply the mapping function to the pixel intensities of the grayscale image
                //Bitmap equalizedBitmap = Bitmap.createBitmap(processed.getWidth(), processed.getHeight(), Bitmap.Config.ARGB_8888);
                //Canvas canvas2 = new Canvas(equalizedBitmap);
                //Paint paint2 = new Paint();
                for (int x_i = 0; x_i < processed.getWidth(); x_i++) {
                    for (int y_i = 0; y_i < processed.getHeight(); y_i++) {
                        int pixel = processed.getPixel(x_i, y_i);
                        int intensity = Color.red(pixel);
                        int equalizedIntensity = mapping[intensity];
                        int equalizedPixel = Color.rgb(equalizedIntensity, equalizedIntensity, equalizedIntensity);
                        //canvas2.drawPoint(x_i, y_i, paint2);
                        processed.setPixel(x_i, y_i, equalizedPixel);
                    }
                }
            }


                Bitmap resized = Bitmap.createScaledBitmap(processed, TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, false);

                //mImageView.setImageBitmap(equalizedBitmap);
                mImageViewProcessed.setImageBitmap(resized);

                if (0==0) {
                    List<SimilarityClassifier.Recognition> results = detector.recognizeImage(resized, false);

                    if (train) {
                        Log.d("NSP debug: ", "Registering face!");

                        int randomNum = -1;

                        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.LOLLIPOP) {
                            randomNum = ThreadLocalRandom.current().nextInt(0, 1000 + 1);
                        }
                        detector.register(Integer.toString(randomNum), detector.recognizeImage(resized, true).get(0));
                        faceGraphic.setId(Integer.toString(randomNum));

                    } else {
                        for (SimilarityClassifier.Recognition rec : results){
                            float distance = rec.getDistance();
                            String title = rec.getTitle();

                            if (distance < 1.1){
                                Log.d("NSP debug: ", rec.getDistance().toString());
                                Log.d("NSP debug: ", rec.getTitle());
                                faceGraphic.setId(rec.getTitle());
                            }else{
                                Log.d("NSP debug: ", "No similar face found!");
                                faceGraphic.setId("Not recognised!");
                            }
                        }

                    }


                }
                //TensorImage.fromBitmap(equalizedBitmap);
           // }
        }

        train = false;

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
