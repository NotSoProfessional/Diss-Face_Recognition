package com.google.codelab.mlkit;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.os.Trace;
import android.util.Log;
import android.widget.Toast;

import com.google.codelab.mlkit.tflite.SimilarityClassifier;
import com.google.codelab.mlkit.tflite.TFLiteObjectDetectionAPIModel;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;

public class Benchmark {

    private static final int TF_OD_API_INPUT_SIZE = 112;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "mobile_face_net.tflite";

    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";

    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;

    private SimilarityClassifier mfnRecog;

    private Activity mActivity;
    private FaceDetector detector;
    private ImagePreProcessor preProcessor;
    private ArrayList<CVFaceRecognizer> cvFR = new ArrayList<>();

    private AssetManager assetManager;

    private boolean training = false;
    private boolean testing = false;
    private List<Face> faces;
    private Bitmap current_image;

    private ArrayList<List<Face>> lfaces = new ArrayList<>();
    private ArrayList<Bitmap> bfaces = new ArrayList<>();

    private ArrayList<Integer> tp = new ArrayList<>();
    private ArrayList<Integer> fp = new ArrayList<>();
    private ArrayList<Integer> fn = new ArrayList<>();
    private ArrayList<Integer> tp_mfn = new ArrayList<>();
    private ArrayList<Integer> fp_mfn = new ArrayList<>();

    private ArrayList<Integer> fn_mfn = new ArrayList<>();
    private String folderName = "lfw100n";

    private ArrayList<Long> alignTimes = new ArrayList<>();
    private ArrayList<Long> grayTimes = new ArrayList<>();
    private ArrayList<Long> equaliseTimes = new ArrayList<>();
    private ArrayList<Long> bilateralTimes = new ArrayList<>();

    private ArrayList<Long> tfInferenceTimes = new ArrayList<>();

    private ArrayList<Long> tfPredictTimes = new ArrayList<>();

    private ArrayList<Long> cvPredictTimes = new ArrayList<>();
    private ArrayList<Long> cvTrainTimes = new ArrayList<>();

    private ArrayList<ArrayList<Long>> groupTimes = new ArrayList<>();
    private ImagePreProcessor.Options ppOptions;
    private boolean mUseGPU;

    public Benchmark(AssetManager assetManager, Activity activity, ImagePreProcessor.Options ppOptions, boolean useNNAPI, boolean useGPU, boolean useXNNPack){
        FaceDetectorOptions options =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                        .build();

        /*ImagePreProcessor.Options ppOptions = new ImagePreProcessor.Options();
        ppOptions.setEqualise(true);
        ppOptions.setGrayscale(true);
        ppOptions.setRotate(true);
        ppOptions.setBilateral(true);
        ppOptions.setUseOpenCV(true);*/

        this.assetManager = assetManager;
        this.ppOptions = ppOptions;
        mUseGPU = useGPU;
        mActivity = activity;

        detector = FaceDetection.getClient(options);
        preProcessor = new ImagePreProcessor(ppOptions, mActivity.getApplicationContext());

        cvFR.add(new CVFaceRecognizer(CVFaceRecognizer.Method.EIGENFACE));
        cvFR.add(new CVFaceRecognizer(CVFaceRecognizer.Method.FISHERFACE));
        cvFR.add(new CVFaceRecognizer(CVFaceRecognizer.Method.LBPH));

        try {
            mfnRecog =
                    TFLiteObjectDetectionAPIModel.create(
                            assetManager,
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED,
                            useNNAPI,
                            useGPU,
                            useXNNPack,
                            2
                    );
        } catch (final IOException e) {
            e.printStackTrace();
            Log.e("Exception initializing classifier!", e.getMessage());
        }

        //org.opencv.core.Core.setNumThreads(1);
    }

    public void Run() {
        // Set the folder name of the subdirectories you want to search for images in

        Reset();


        if (mUseGPU){
            mfnRecog.ReInitModel(assetManager, TF_OD_API_MODEL_FILE, false, true, false, 4);
        }

        Train();

        Test();

        GetScore();

            Log.d("NSP debug", "FINAL COUNT: " + Integer.toString(bfaces.size()));


    }

    public void RunGroup(){
        folderName = "group_photos";

        Reset();
        TrainGroup();
        GetGroupTimes();
        GetScore();
    }

    private void Reset(){
        cvFR.get(2).clearFaces();
        fp.clear();
        fn.clear();
        tp.clear();
        fp_mfn.clear();
        fn_mfn.clear();
        tp_mfn.clear();

        tfInferenceTimes.clear();
        tfPredictTimes.clear();
        bilateralTimes.clear();
        equaliseTimes.clear();
        grayTimes.clear();
        alignTimes.clear();
    }

    public void GetGroupTimes(){
        ArrayList<Long> timesAvg = new ArrayList<>();

        for (int i = 0; i < groupTimes.size(); i++){
            ArrayList<Long> times = groupTimes.get(i);
            timesAvg.add(0L);
            for (Long time : times){
                timesAvg.set(i, timesAvg.get(i) + time);
            }

            timesAvg.set(i, timesAvg.get(i) / times.size());
        }

        for (long time : timesAvg){
            Log.d("NSP debug", "Group time: " + Long.toString(time) + " [ns]");
        }
    }

    public void GetScore(){

        String[] subDirectories = new String[0];

        int tp_total = 0;
        int fp_total = 0;
        int fn_total = 0;
        float precision = 0;
        float recall = 0;
        float f1score = 0;

        for (int i = 0; i < tp.size(); i++) {
            tp_total += tp.get(i);
            fp_total += fp.get(i);
            fn_total += fn.get(i);
        }

        int tp_total_mfn = 0;
        int fp_total_mfn = 0;
        int fn_total_mfn = 0;
        float precision_mfn = 0;
        float recall_mfn = 0;
        float f1score_mfn = 0;

        for (int i = 0; i < tp.size(); i++) {
            tp_total_mfn += tp_mfn.get(i);
            fp_total_mfn += fp_mfn.get(i);
            fn_total_mfn += fn_mfn.get(i);
        }


        precision = (float) tp_total/(float) (tp_total+fp_total);
        recall = (float) tp_total/(float) (tp_total+fn_total);
        f1score = (2 * (precision * recall)) / (precision + recall);

        precision_mfn = (float) tp_total_mfn/(float) (tp_total_mfn+fp_total_mfn);
        recall_mfn = (float) tp_total_mfn / (float) (tp_total_mfn+fn_total_mfn);
        f1score_mfn = (2 * (precision_mfn * recall_mfn)) / (precision_mfn + recall_mfn);

        long align_time = 0;
        long gray_time = 0;
        long equalise_time = 0;
        long bilateral_time = 0;

        for (Long time : alignTimes) {align_time+=time;}
        for (Long time : grayTimes) {gray_time+=time;}
        for (Long time : equaliseTimes) {equalise_time+=time;}
        for (Long time : bilateralTimes) {bilateral_time+=time;}

        align_time /= alignTimes.size();
        gray_time /= grayTimes.size();
        equalise_time /= equaliseTimes.size();
        bilateral_time /= bilateralTimes.size();

        Log.d("NSP debug", "| Precision MFN: " + Float.toString(precision_mfn)
                + ", " + "Recall MFN: " + Float.toString(recall_mfn)
                + ", " + "F1 Score MFN: " + Float.toString(f1score_mfn));

        Log.d("NSP debug", "| Precision LBPH: " + Float.toString(precision)
                + ", " + "Recall LBPH: " + Float.toString(recall)
                + ", " + "F1 Score LBPH: " + Float.toString(f1score));

        Log.d("NSP debug", "Avg. Align time: " + Long.toString(align_time) + " [ns]");
        Log.d("NSP debug", "Avg. Gray time: " + Long.toString(gray_time) + " [ns]");
        Log.d("NSP debug", "Avg. Equalise time: " + Long.toString(equalise_time) + " [ns]");
        Log.d("NSP debug", "Avg. Bilateral time: " + Long.toString(bilateral_time) + " [ns]");
        Log.d("NSP debug", "Avg. Total Pre-Process: " + Long.toString(align_time+gray_time+equalise_time+bilateral_time) + " [ns]");

        long inferenceTimeMFN = 0;
        long inferenceTimeTotal = 0;
        long predictTimeMFN = 0;
        long predictTimeTotal = 0;

        for (int i = 0; i < tfInferenceTimes.size(); i++) {
            inferenceTimeTotal += tfInferenceTimes.get(i);
            predictTimeTotal += tfPredictTimes.get(i);
        }

        if (tfInferenceTimes.size() > 0){
            inferenceTimeMFN = inferenceTimeTotal/tfInferenceTimes.size();
            predictTimeMFN = predictTimeTotal/ tfInferenceTimes.size();
        }

        Log.d("NSP debug", "Avg. MFN Inference Time: " + Long.toString(inferenceTimeMFN) + " [ns]"
                + ", Avg. MFN Predict Time: " + Long.toString(predictTimeMFN));

        long predictTimeCV = 0;
        long predictTimeCVTotal = 0;
        long trainTimeCV = 0;
        long trainTimeCVTotal = 0;

        for (int i = 0; i < cvPredictTimes.size(); i++) {
            predictTimeCVTotal += cvPredictTimes.get(i);
        }

        for (int i = 0; i < cvTrainTimes.size(); i++) {
            trainTimeCVTotal += cvTrainTimes.get(i);
        }


        if (cvPredictTimes.size() > 0){
            predictTimeCV = predictTimeCVTotal/cvPredictTimes.size();
            trainTimeCV = trainTimeCVTotal/ cvPredictTimes.size();
        }

        Log.d("NSP debug", "Avg. CV Predict Time: " + Long.toString(predictTimeCV) + " [ns]"
                + ", Avg. CV Train Time: " + Long.toString(trainTimeCV));
    }

    private void DetectFace(Bitmap image, CountDownLatch countDownLatch, int label) {
        try {
            InputImage inputImage = InputImage.fromBitmap(image, 0);
            detector.process(inputImage)
                    .addOnSuccessListener(faces -> {
                        if (faces.size() > 0) {
                            lfaces.set(label, faces);
                        }
                        countDownLatch.countDown();
                    })
                    .addOnFailureListener(e -> {
                        // Handle the error
                        // Count down the latch even if there's an error
                        countDownLatch.countDown();
                    });
        } catch (Exception e) {
            e.printStackTrace();
            // Count down the latch even if there's an error
            countDownLatch.countDown();
        }
    }

    private void TrainGroup(){
        try {
            int label = 0;

            // Get a list of all subdirectories in the specified folder
            String[] subDirectories = assetManager.list(folderName);

            for (String subDirectory : subDirectories) {
                // Get a list of all image files in the subdirectory
                String[] imageFiles = assetManager.list(folderName + "/" + subDirectory);

                // If there are image files in the subdirectory, get the first one and do something with it
                groupTimes.add(new ArrayList<Long>());
                lfaces.add(null);

                if (imageFiles.length > 0) {
                        // Get a reference to the first image file
                        for (int i = 0; i < imageFiles.length; i++) {
                            String firstImageFile = imageFiles[i];

                            InputStream inputStream = assetManager.open(folderName + "/" + subDirectory + "/" + firstImageFile);
                            Bitmap image = BitmapFactory.decodeStream(inputStream);

                            CountDownLatch countDownLatch = new CountDownLatch(1);
                            DetectFace(image, countDownLatch, label);
                            countDownLatch.await();

                            ArrayList<Bitmap> processedFaces = new ArrayList<>();

                            long startTime = System.nanoTime();

                            for (Face face : lfaces.get(label)) {

                                if (true==true){
                                    new Thread(new Runnable() {
                                        @Override
                                        public void run() {
                                            processedFaces.add(preProcessor.Process(face, image));
                                            getPreProcessTimes();
                                        }
                                    }).run();
                                } else {
                                    processedFaces.add(preProcessor.Process(face, image));
                                    getPreProcessTimes();
                                }

                            }

                            while (processedFaces.size() < lfaces.get(label).size()) ;

                            long endTime = System.nanoTime();
                            groupTimes.get(label).add(endTime - startTime);

                            /*for (int j = 0; j < processedFaces.size(); j++) {

                                Bitmap processed = processedFaces.get(j);

                                Trace.beginSection("LBPH Add Face");
                                cvFR.get(2).addFace(processed, label);
                                Trace.endSection();

                                Trace.beginSection("MFN Register Face");
                                mfnRecog.register(Integer.toString(label), mfnRecog.recognizeImage(processed, true).get(0));
                                Trace.endSection();
                                tfInferenceTimes.add(mfnRecog.getInferenceTime());
                                tfPredictTimes.add(mfnRecog.getPredictTime());
                            }*/

                        }
                    label++;
                }
            }

            //cvFR.get(2).train();

        } catch (IOException | InterruptedException e) {
            // Handle any errors that occur when accessing the assets
            e.printStackTrace();
        }
    }


    private void Train(){

        try {
            int label = 0;

            // Get a list of all subdirectories in the specified folder
            String[] subDirectories = assetManager.list(folderName);

            // Initialize a CountDownLatch with the number of subdirectories
            CountDownLatch countDownLatch = new CountDownLatch(subDirectories.length);

            // Loop through each subdirectory and get the first image file
            for (String subDirectory : subDirectories) {
                // Get a list of all image files in the subdirectory
                String[] imageFiles = assetManager.list(folderName + "/" + subDirectory);

                lfaces.add(null);

                // If there are image files in the subdirectory, get the first one and do something with it
                if (imageFiles.length > 0) {
                    // Get a reference to the first image file
                    String firstImageFile = imageFiles[0];

                    InputStream inputStream = assetManager.open(folderName + "/" + subDirectory + "/" + firstImageFile);
                    Bitmap image = BitmapFactory.decodeStream(inputStream);
                    inputStream.close();

                    DetectFace(image, countDownLatch, label);
                    label++;
                }
            }

            // Wait for all the faces to be processed before logging the count
            countDownLatch.await();

            label = 0;

            for (String subDirectory : subDirectories) {
                // Get a list of all image files in the subdirectory
                String[] imageFiles = assetManager.list(folderName + "/" + subDirectory);

                // If there are image files in the subdirectory, get the first one and do something with it
                if (imageFiles.length > 0) {
                    if (lfaces.get(label) != null) {
                        // Get a reference to the first image file
                        String firstImageFile = imageFiles[0];

                        InputStream inputStream = assetManager.open(folderName + "/" + subDirectory + "/" + firstImageFile);
                        Bitmap image = BitmapFactory.decodeStream(inputStream);

                        //for (Face face : lfaces.get(label)){
                            Face face = lfaces.get(label).get(0);

                            Bitmap processed = preProcessor.Process(face, image);

                            getPreProcessTimes();

                            Trace.beginSection("LBPH Add Face");
                            cvFR.get(2).addFace(processed, label);
                            Trace.endSection();

                        Trace.beginSection("MFN Register Face");
                            mfnRecog.register(Integer.toString(label), mfnRecog.recognizeImage(processed, true).get(0));
                        Trace.endSection();
                            tfInferenceTimes.add(mfnRecog.getInferenceTime());
                            tfPredictTimes.add(mfnRecog.getPredictTime());
                            //bfaces.add(processed);
                        //}

                    }
                    label++;
                }
            }

            cvFR.get(2).train();
            cvTrainTimes.add(cvFR.get(2).getTrainTime());

        } catch (IOException | InterruptedException e) {
            // Handle any errors that occur when accessing the assets
            e.printStackTrace();
        }

    }

    private void Test(){

        try {
            int label = 0;

            // Get a list of all subdirectories in the specified folder
            String[] subDirectories = assetManager.list(folderName);

            label = 0;

            for (String subDirectory : subDirectories) {
                // Get a list of all image files in the subdirectory
                String[] imageFiles = assetManager.list(folderName + "/" + subDirectory);

                if (label == 0){
                    for (int i = 0; i < subDirectories.length; i++) {
                        tp.add(0);
                        fp.add(0);
                        fn.add(0);
                        tp_mfn.add(0);
                        fp_mfn.add(0);
                        fn_mfn.add(0);
                    }
                }

                // If there are image files in the subdirectory, get the first one and do something with it
                if (imageFiles.length > 0) {
                    if (lfaces.get(label) != null) {
                        // Get a reference to the first image file
                        for (int i = 1; i < imageFiles.length; i++){

                            InputStream inputStream = assetManager.open(folderName + "/" + subDirectory + "/" + imageFiles[i]);
                            Bitmap image = BitmapFactory.decodeStream(inputStream);

                            CountDownLatch countDownLatch = new CountDownLatch(1);
                            DetectFace(image, countDownLatch, label);
                            countDownLatch.await();
                            //for (Face face : lfaces.get(label)){
                                Face face = lfaces.get(label).get(0);

                                Bitmap processed = preProcessor.Process(face, image);

                                getPreProcessTimes();

                            Trace.beginSection("LBPH Predict Face");
                                CVFaceRecognizer.Result result = cvFR.get(2).predict(processed);
                                cvPredictTimes.add(cvFR.get(2).getPredictTime());
                                Trace.endSection();

                            Trace.beginSection("MFN Predict Face");
                                List<SimilarityClassifier.Recognition> results = mfnRecog.recognizeImage(processed, false);
                            Trace.endSection();
                                tfInferenceTimes.add(mfnRecog.getInferenceTime());
                                tfPredictTimes.add(mfnRecog.getPredictTime());

                                if ((result.confidence < 6000 & result.confidence > 4800) | (result.confidence < 100 & result.confidence > 50) ) {
                                    if (result.label == label){
                                        tp.set(label, tp.get(label) + 1);
                                    }else{
                                        fp.set(result.label, fp.get(result.label) + 1);
                                    }

                                } else {
                                    fn.set(label, fn.get(label)+1);
                                }

                                SimilarityClassifier.Recognition rec = results.get(0);

                                if (rec.getDistance() < 1.1){
                                    if (rec.getTitle() == Integer.toString(label)){
                                        tp_mfn.set(label, tp_mfn.get(label) + 1);
                                    }else{
                                        fp_mfn.set(Integer.parseInt(rec.getTitle()), fp_mfn.get(Integer.parseInt(rec.getTitle())) + 1);
                                    }
                                } else {
                                    fn_mfn.set(label, fn_mfn.get(label)+1);
                                }
                            //}
                        }
                    }
                    label++;
                }
            }

        } catch (IOException | InterruptedException e) {
            // Handle any errors that occur when accessing the assets
            e.printStackTrace();
        }

    }

    private void getPreProcessTimes(){
        alignTimes.add(preProcessor.getAlignTime());
        grayTimes.add(preProcessor.getGrayTime());
        equaliseTimes.add(preProcessor.getEqualiseTime());
        bilateralTimes.add(preProcessor.getBilateralTime());
    }

    private void clearPreProcessTimes(){
        alignTimes.clear();
        grayTimes.clear();
        equaliseTimes.clear();
        bilateralTimes.clear();
    }

}
